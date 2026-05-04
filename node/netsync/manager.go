// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"container/list"
	"fmt"
	"math/big"
	"math/rand"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/database"
	"github.com/pearl-research-labs/pearl/node/mempool"
	peerpkg "github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	// minInFlightBlocks is the minimum number of blocks that should be
	// in the request queue for headers-first mode before requesting
	// more.
	minInFlightBlocks = 10

	// maxRejectedTxns is the maximum number of rejected transactions
	// hashes to store in memory.
	maxRejectedTxns = 1000

	// maxRequestedBlocks is the maximum number of requested block
	// hashes to store in memory.
	maxRequestedBlocks = wire.MaxInvPerMsg

	// maxRequestedTxns is the maximum number of requested transactions
	// hashes to store in memory.
	maxRequestedTxns = wire.MaxInvPerMsg

	// maxStallDuration is the time after which we will disconnect our
	// current sync peer if we haven't made progress.
	maxStallDuration = 3 * time.Minute

	// stallSampleInterval the interval at which we will check to see if our
	// sync has stalled.
	stallSampleInterval = 30 * time.Second

	// headersResponseTime is the minimum interval between getheaders
	// messages to the same peer. Matches HEADERS_RESPONSE_TIME in
	// Bitcoin Core's net_processing.cpp.
	headersResponseTime = 2 * time.Minute

	// peerQualityThreshold is the number of non-near-tip announcements
	// tolerated before a peer reverts from high-quality to low-quality.
	// Peers start low-quality (counter = threshold). A block that becomes
	// the new tip resets the counter to 0 (high-quality). Each non-near-tip
	// block/header increments the counter.
	peerQualityThreshold = 5

	// antiDoSBufferBlocks is the number of blocks of work subtracted from
	// the tip's cumulative work to compute the anti-DoS threshold. Forks
	// within this many blocks of the tip bypass presync.
	antiDoSBufferBlocks int64 = 100

	// redownloadPendingCap bounds the Tier-2 REDOWNLOAD buffer: the
	// in-order FIFO of approved entries either awaiting block arrival
	// (getdata in flight) or buffered pending predecessor acceptance.
	redownloadPendingCap = 100

	// syncPeerCooldown is how long an address that stalled as syncnode
	// is excluded from re-selection.
	syncPeerCooldown = 10 * time.Minute
)

// zeroHash is the zero value hash (all zeros).  It is defined as a convenience.
var zeroHash chainhash.Hash

// newPeerMsg signifies a newly connected peer to the block handler.
type newPeerMsg struct {
	peer *peerpkg.Peer
}

// blockMsg packages a block message and the peer it came from together
// so the block handler has access to that information.
type blockMsg struct {
	block *btcutil.Block
	peer  *peerpkg.Peer
	reply chan error
}

// invMsg packages an inv message and the peer it came from together
// so the block handler has access to that information.
type invMsg struct {
	inv  *wire.MsgInv
	peer *peerpkg.Peer
}

// PeerVerdict carries the result of header validation for a peer so the
// server layer can decide on punishment (ban or disconnect).
type PeerVerdict struct {
	PeerID int32
	Err    error
}

// headersMsg packages a headers message and the peer it came from
// together so the block handler has access to that information.
type headersMsg struct {
	headers *wire.MsgHeaders
	peer    *peerpkg.Peer
}

// notFoundMsg packages a notfound message and the peer it came from
// together so the block handler has access to that information.
type notFoundMsg struct {
	notFound *wire.MsgNotFound
	peer     *peerpkg.Peer
}

// donePeerMsg signifies a newly disconnected peer to the block handler.
type donePeerMsg struct {
	peer *peerpkg.Peer
}

// txMsg packages a tx message and the peer it came from together
// so the block handler has access to that information.
type txMsg struct {
	tx    *btcutil.Tx
	peer  *peerpkg.Peer
	reply chan struct{}
}

// getSyncPeerMsg is a message type to be sent across the message channel for
// retrieving the current sync peer.
type getSyncPeerMsg struct {
	reply chan int32
}

// processBlockResponse is a response sent to the reply channel of a
// processBlockMsg.
type processBlockResponse struct {
	isOrphan bool
	err      error
}

// processBlockMsg is a message type to be sent across the message channel
// for requested a block is processed.  Note this call differs from blockMsg
// above in that blockMsg is intended for blocks that came from peers and have
// extra handling whereas this message essentially is just a concurrent safe
// way to call ProcessBlock on the internal block chain instance.
type processBlockMsg struct {
	block *btcutil.Block
	flags blockchain.BehaviorFlags
	reply chan processBlockResponse
}

// isCurrentMsg is a message type to be sent across the message channel for
// requesting whether or not the sync manager believes it is synced with the
// currently connected peers.
type isCurrentMsg struct {
	reply chan bool
}

// pauseMsg is a message type to be sent across the message channel for
// pausing the sync manager.  This effectively provides the caller with
// exclusive access over the manager until a receive is performed on the
// unpause channel.
type pauseMsg struct {
	unpause <-chan struct{}
}

// headerNode is used as a node in a list of headers that are linked together
// between checkpoints.
type headerNode struct {
	height int32
	hash   *chainhash.Hash
}

// peerSyncState stores additional information that the SyncManager tracks
// about a peer.
type peerSyncState struct {
	syncCandidate   bool
	requestQueue    []*wire.InvVect
	requestedTxns   map[chainhash.Hash]struct{}
	requestedBlocks map[chainhash.Hash]struct{}

	// headersSyncState tracks an active presync session with this peer.
	// Non-nil only while the peer is undergoing the two-phase presync.
	headersSyncState *HeadersSyncState

	// lastGetHeadersTime is when we last sent getheaders to this peer.
	// Used for rate-limiting (one getheaders per headersResponseTime).
	lastGetHeadersTime time.Time

	// peerQualityCounter tracks whether this peer provides near-tip data.
	// Starts at peerQualityThreshold (low-quality). Reset to 0 when the
	// peer sends a block that becomes the new chain tip (high-quality).
	// Incremented when the peer sends non-near-tip blocks or headers.
	// While < peerQualityThreshold, inv messages get direct getdata;
	// otherwise they go through getheaders first.
	peerQualityCounter int

	// pipelinedLocatorHead is the tip hash used as the first locator entry
	// of an outstanding speculative GETHEADERS. Non-nil means there is one
	// speculative request in flight.
	pipelinedLocatorHead *chainhash.Hash

	// redownloadExpected is the Tier-2 in-order FIFO of REDOWNLOAD entries
	// for which getdata has been sent or block has arrived pending acceptance.
	redownloadExpected []ApprovedRedownloadEntry

	// redownloadPendingBlocks holds blocks that arrived but whose predecessor
	// has not yet been accepted. Members are a subset of redownloadExpected.
	redownloadPendingBlocks map[chainhash.Hash]*btcutil.Block
}

// clearHeadersRateLimit releases the per-peer getheaders rate limit.
func (state *peerSyncState) clearHeadersRateLimit() {
	state.lastGetHeadersTime = time.Time{}
}

func isPeerHighQuality(state *peerSyncState) bool {
	return state.peerQualityCounter < peerQualityThreshold
}

// limitAdd is a helper function for maps that require a maximum limit by
// evicting a random value if adding the new value would cause it to
// overflow the maximum allowed.
func limitAdd(m map[chainhash.Hash]struct{}, hash chainhash.Hash, limit int) {
	if len(m)+1 > limit {
		// Remove a random entry from the map.  For most compilers, Go's
		// range statement iterates starting at a random item although
		// that is not 100% guaranteed by the spec.  The iteration order
		// is not important here because an adversary would have to be
		// able to pull off preimage attacks on the hashing function in
		// order to target eviction of specific entries anyways.
		for txHash := range m {
			delete(m, txHash)
			break
		}
	}
	m[hash] = struct{}{}
}

// SyncManager is used to communicate block related messages with peers. The
// SyncManager is started as by executing Start() in a goroutine. Once started,
// it selects peers to sync from and starts the initial block download. Once the
// chain is in sync, the SyncManager handles incoming block and header
// notifications and relays announcements of new blocks to peers.
type SyncManager struct {
	peerNotifier   PeerNotifier
	started        int32
	shutdown       int32
	chain          *blockchain.BlockChain
	txMemPool      *mempool.TxPool
	chainParams    *chaincfg.Params
	progressLogger *blockProgressLogger
	msgChan        chan interface{}
	peerVerdicts   chan PeerVerdict
	wg             sync.WaitGroup
	quit           chan struct{}

	// These fields should only be accessed from the blockHandler thread
	rejectedTxns     map[chainhash.Hash]struct{}
	requestedTxns    map[chainhash.Hash]struct{}
	requestedBlocks  map[chainhash.Hash]struct{}
	syncPeer         *peerpkg.Peer
	peerStates       map[*peerpkg.Peer]*peerSyncState
	lastProgressTime time.Time

	// The following fields are used for headers-first mode.
	headersFirstMode bool
	headerList       *list.List
	startHeader      *list.Element
	nextCheckpoint   *chaincfg.Checkpoint

	// An optional fee estimator.
	feeEstimator *mempool.FeeEstimator

	// recentlyFailedSync tracks outbound peer addresses that stalled
	// while serving as syncnode. pickSyncCandidate skips entries
	// within syncPeerCooldown and lazy-evicts expired ones.
	recentlyFailedSync map[string]time.Time
}

// resetHeaderState sets the headers-first mode state to values appropriate for
// syncing from a new peer.
func (sm *SyncManager) resetHeaderState(newestHash *chainhash.Hash, newestHeight int32) {
	sm.headersFirstMode = false
	sm.headerList.Init()
	sm.startHeader = nil

	// When there is a next checkpoint, add an entry for the latest known
	// block into the header pool.  This allows the next downloaded header
	// to prove it links to the chain properly.
	if sm.nextCheckpoint != nil {
		node := headerNode{height: newestHeight, hash: newestHash}
		sm.headerList.PushBack(&node)
	}
}

// findNextHeaderCheckpoint returns the next checkpoint after the passed height.
// It returns nil when there is not one either because the height is already
// later than the final checkpoint or some other reason such as disabled
// checkpoints.
func (sm *SyncManager) findNextHeaderCheckpoint(height int32) *chaincfg.Checkpoint {
	checkpoints := sm.chain.Checkpoints()
	if len(checkpoints) == 0 {
		return nil
	}

	// There is no next checkpoint if the height is already after the final
	// checkpoint.
	finalCheckpoint := &checkpoints[len(checkpoints)-1]
	if height >= finalCheckpoint.Height {
		return nil
	}

	// Find the next checkpoint.
	nextCheckpoint := finalCheckpoint
	for i := len(checkpoints) - 2; i >= 0; i-- {
		if height >= checkpoints[i].Height {
			break
		}
		nextCheckpoint = &checkpoints[i]
	}
	return nextCheckpoint
}

// pickSyncCandidate returns a random sync-peer candidate, filtered by:
//   - state.syncCandidate (SFNodeNetwork / pruned-with-block-threshold,
//     set at handshake by isSyncCandidate).
//   - Outbound only. peer.LastBlock() from the version handshake is
//     unauthenticated; restricting to outbound peers limits candidates
//     to addresses we chose to dial (addr.dat / dnsseed).
//   - recentlyFailedSync cooldown to prevent immediate re-selection of
//     peers that previously stalled as syncnode.
//
// peer.LastBlock() is intentionally not used for ranking. Mirrors Bitcoin
// Core's CanServeBlocks + fPreferredDownload posture (net_processing.cpp).
func (sm *SyncManager) pickSyncCandidate() *peerpkg.Peer {
	now := time.Now()
	var candidates []*peerpkg.Peer
	for peer, state := range sm.peerStates {
		if !state.syncCandidate {
			continue
		}
		if peer.Inbound() {
			continue
		}
		if t, ok := sm.recentlyFailedSync[peer.Addr()]; ok {
			if now.Sub(t) < syncPeerCooldown {
				continue
			}
			delete(sm.recentlyFailedSync, peer.Addr())
		}
		candidates = append(candidates, peer)
	}

	if len(candidates) == 0 {
		if sm.chain.IsCurrent() {
			best := sm.chain.BestSnapshot()
			log.Infof("Caught up to block %s(%d)",
				best.Hash.String(), best.Height)
		}
		return nil
	}
	return candidates[rand.Intn(len(candidates))]
}

// startSync will choose the best peer among the available candidate peers to
// download/sync the blockchain from.  When syncing is already running, it
// simply returns.
func (sm *SyncManager) startSync() {
	// Return now if we're already syncing.
	if sm.syncPeer != nil {
		return
	}

	best := sm.chain.BestSnapshot()
	bestPeer := sm.pickSyncCandidate()

	// Start syncing from the best peer if one was selected.
	if bestPeer != nil {
		// Clear the requestedBlocks if the sync peer changes, otherwise
		// we may ignore blocks we need that the last sync peer failed
		// to send.
		sm.requestedBlocks = make(map[chainhash.Hash]struct{})

		locator, err := sm.chain.LatestBlockLocator()
		if err != nil {
			log.Errorf("Failed to get block locator for the "+
				"latest block: %v", err)
			return
		}

		log.Infof("Syncing to block height %d from peer %v",
			bestPeer.LastBlock(), bestPeer.Addr())

		// Request headers from the sync peer so we learn about its
		// chain. Only enter checkpoint-based headersFirstMode when
		// there is an actual checkpoint to sync against; presync
		// handles its own headers-first logic independently.
		stopHash := &zeroHash
		if sm.nextCheckpoint != nil && best.Height < sm.nextCheckpoint.Height {
			stopHash = sm.nextCheckpoint.Hash
			sm.headersFirstMode = true
		}
		if bpState, ok := sm.peerStates[bestPeer]; ok {
			_ = sm.pushGetHeadersDirect(bestPeer, bpState, locator, stopHash, true)
		} else {
			bestPeer.PushGetHeadersMsg(locator, stopHash, true)
		}
		log.Infof("Downloading headers from peer %s (current height %d)",
			bestPeer.Addr(), best.Height)
		sm.syncPeer = bestPeer

		// Reset the last progress time now that we have a non-nil
		// syncPeer to avoid instantly detecting it as stalled in the
		// event the progress time hasn't been updated recently.
		sm.lastProgressTime = time.Now()
	} else {
		log.Warnf("No sync peer candidates available")
	}
}

// isSyncCandidate returns whether or not the peer is a candidate to consider
// syncing from.
func (sm *SyncManager) isSyncCandidate(peer *peerpkg.Peer) bool {
	// Typically a peer is not a candidate for sync if it's not a full node,
	// however regression test is special in that the regression tool is
	// not a full node and still needs to be considered a sync candidate.
	if sm.chainParams == &chaincfg.RegressionNetParams {
		// The peer is not a candidate if it's not coming from localhost
		// or the hostname can't be determined for some reason.
		host, _, err := net.SplitHostPort(peer.Addr())
		if err != nil {
			return false
		}

		if host != "127.0.0.1" && host != "localhost" {
			return false
		}

		// Candidate if all checks passed.
		return true
	}

	var (
		nodeServices = peer.Services()
		fullNode     = nodeServices.HasFlag(wire.SFNodeNetwork)
		prunedNode   = nodeServices.HasFlag(wire.SFNodeNetworkLimited)
	)

	switch {
	case fullNode:
		// Node is a sync candidate if it has all the blocks.

	case prunedNode:
		// Even if the peer is pruned, if they have the node network
		// limited flag, they are able to serve 2 days worth of blocks
		// from the current tip. Therefore, check if our chaintip is
		// within that range.
		bestHeight := sm.chain.BestSnapshot().Height
		peerLastBlock := peer.LastBlock()

		// bestHeight+1 as we need the peer to serve us the next block,
		// not the one we already have.
		if bestHeight+1 <=
			peerLastBlock-wire.NodeNetworkLimitedBlockThreshold {

			return false
		}

	default:
		// If the peer isn't an archival node, and it's not signaling
		// NODE_NETWORK_LIMITED, we can't sync off of this node.
		return false
	}

	// Candidate if all checks passed.
	return true
}

// handleNewPeerMsg deals with new peers that have signalled they may
// be considered as a sync peer (they have already successfully negotiated).  It
// also starts syncing if needed.  It is invoked from the syncHandler goroutine.
func (sm *SyncManager) handleNewPeerMsg(peer *peerpkg.Peer) {
	// Ignore if in the process of shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	log.Infof("New valid peer %s (%s)", peer, peer.UserAgent())

	// Initialize the peer state.
	isSyncCandidate := sm.isSyncCandidate(peer)
	sm.peerStates[peer] = &peerSyncState{
		syncCandidate:      isSyncCandidate,
		requestedTxns:      make(map[chainhash.Hash]struct{}),
		requestedBlocks:    make(map[chainhash.Hash]struct{}),
		peerQualityCounter: peerQualityThreshold,
	}

	// Start syncing by choosing the best candidate if needed.
	if isSyncCandidate && sm.syncPeer == nil {
		sm.startSync()
	}
}

// handleStallSample will switch to a new sync peer if the current one has
// stalled. This is detected when by comparing the last progress timestamp with
// the current time, and disconnecting the peer if we stalled before reaching
// their highest advertised block.
func (sm *SyncManager) handleStallSample() {
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	// No syncpeer — retry selection periodically so cooled-down
	// candidates get re-evaluated as their entries expire.
	if sm.syncPeer == nil {
		if !sm.chain.IsCurrent() {
			sm.startSync()
		}
		return
	}

	// If the stall timeout has not elapsed, exit early.
	if time.Since(sm.lastProgressTime) <= maxStallDuration {
		return
	}

	// Check to see that the peer's sync state exists.
	state, exists := sm.peerStates[sm.syncPeer]
	if !exists {
		return
	}

	sm.clearRequestedState(state)

	// Temporarily exclude the stalled peer from sync-peer selection.
	// Inbound peers are skipped (ephemeral source port, already
	// excluded from candidates).
	if !sm.syncPeer.Inbound() {
		sm.recentlyFailedSync[sm.syncPeer.Addr()] = time.Now()
	}

	disconnectSyncPeer := sm.shouldDCStalledSyncPeer()
	sm.updateSyncPeer(disconnectSyncPeer)

	// Abort stale presync sessions on any peer. A non-zero
	// lastGetHeadersTime means a request is in-flight; if it stays
	// unanswered beyond headersResponseTime the session is dead.
	now := time.Now()
	for peer, st := range sm.peerStates {
		if st.headersSyncState != nil &&
			!st.lastGetHeadersTime.IsZero() &&
			now.Sub(st.lastGetHeadersTime) > headersResponseTime {
			log.Infof("Presync session with peer %s stalled, aborting",
				peer.Addr())
			st.headersSyncState = nil
			st.pipelinedLocatorHead = nil
			st.peerQualityCounter = peerQualityThreshold + 1
		}
	}
}

// shouldDCStalledSyncPeer determines whether or not we should disconnect a
// stalled sync peer. If the peer has stalled and its reported height is greater
// than our own best height, we will disconnect it. Otherwise, we will keep the
// peer connected in case we are already at tip.
func (sm *SyncManager) shouldDCStalledSyncPeer() bool {
	lastBlock := sm.syncPeer.LastBlock()
	startHeight := sm.syncPeer.StartingHeight()

	var peerHeight int32
	if lastBlock > startHeight {
		peerHeight = lastBlock
	} else {
		peerHeight = startHeight
	}

	// If we've stalled out yet the sync peer reports having more blocks for
	// us we will disconnect them. This allows us at tip to not disconnect
	// peers when we are equal or they temporarily lag behind us.
	best := sm.chain.BestSnapshot()
	return peerHeight > best.Height
}

// handleDonePeerMsg deals with peers that have signalled they are done.  It
// removes the peer as a candidate for syncing and in the case where it was
// the current sync peer, attempts to select a new best peer to sync from.  It
// is invoked from the syncHandler goroutine.
func (sm *SyncManager) handleDonePeerMsg(peer *peerpkg.Peer) {
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received done peer message for unknown peer %s", peer)
		return
	}

	// Remove the peer from the list of candidate peers.
	delete(sm.peerStates, peer)

	log.Infof("Lost peer %s", peer)

	sm.clearRequestedState(state)

	if peer == sm.syncPeer {
		// Update the sync peer. The server has already disconnected the
		// peer before signaling to the sync manager.
		sm.updateSyncPeer(false)
	}
}

// clearRequestedState wipes all expected transactions and blocks from the sync
// manager's requested maps that were requested under a peer's sync state, This
// allows them to be rerequested by a subsequent sync peer.
func (sm *SyncManager) clearRequestedState(state *peerSyncState) {
	// Remove requested transactions from the global map so that they will
	// be fetched from elsewhere next time we get an inv.
	for txHash := range state.requestedTxns {
		delete(sm.requestedTxns, txHash)
	}

	// Remove requested blocks from the global map so that they will be
	// fetched from elsewhere next time we get an inv.
	// TODO: we could possibly here check which peers have these blocks
	// and request them now to speed things up a little.
	for blockHash := range state.requestedBlocks {
		delete(sm.requestedBlocks, blockHash)
	}
}

// updateSyncPeer choose a new sync peer to replace the current one. If
// dcSyncPeer is true, this method will also disconnect the current sync peer.
// If we are in header first mode, any header state related to prefetching is
// also reset in preparation for the next sync peer.
func (sm *SyncManager) updateSyncPeer(dcSyncPeer bool) {
	log.Debugf("Updating sync peer, no progress for: %v",
		time.Since(sm.lastProgressTime))

	// First, disconnect the current sync peer if requested.
	if dcSyncPeer {
		sm.syncPeer.Disconnect()
	}

	// Reset any header state before we choose our next active sync peer.
	if sm.headersFirstMode {
		best := sm.chain.BestSnapshot()
		sm.resetHeaderState(&best.Hash, best.Height)
	}

	sm.syncPeer = nil
	sm.startSync()
}

// handleTxMsg handles transaction messages from all peers.
func (sm *SyncManager) handleTxMsg(tmsg *txMsg) {
	peer := tmsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received tx message from unknown peer %s", peer)
		return
	}

	// NOTE:  BitcoinJ, and possibly other wallets, don't follow the spec of
	// sending an inventory message and allowing the remote peer to decide
	// whether or not they want to request the transaction via a getdata
	// message.  Unfortunately, the reference implementation permits
	// unrequested data, so it has allowed wallets that don't follow the
	// spec to proliferate.  While this is not ideal, there is no check here
	// to disconnect peers for sending unsolicited transactions to provide
	// interoperability.
	txHash := tmsg.tx.Hash()

	// Ignore transactions that we have already rejected.  Do not
	// send a reject message here because if the transaction was already
	// rejected, the transaction was unsolicited.
	if _, exists = sm.rejectedTxns[*txHash]; exists {
		log.Debugf("Ignoring unsolicited previously rejected "+
			"transaction %v from %s", txHash, peer)
		return
	}

	// Process the transaction to include validation, insertion in the
	// memory pool, orphan handling, etc.
	acceptedTxs, err := sm.txMemPool.ProcessTransaction(tmsg.tx,
		true, true, mempool.Tag(peer.ID()))

	// Remove transaction from request maps. Either the mempool/chain
	// already knows about it and as such we shouldn't have any more
	// instances of trying to fetch it, or we failed to insert and thus
	// we'll retry next time we get an inv.
	delete(state.requestedTxns, *txHash)
	delete(sm.requestedTxns, *txHash)

	if err != nil {
		// Do not request this transaction again until a new block
		// has been processed.
		limitAdd(sm.rejectedTxns, *txHash, maxRejectedTxns)

		// When the error is a rule error, it means the transaction was
		// simply rejected as opposed to something actually going wrong,
		// so log it as such.  Otherwise, something really did go wrong,
		// so log it as an actual error.
		if _, ok := err.(mempool.RuleError); ok {
			log.Debugf("Rejected transaction %v from %s: %v",
				txHash, peer, err)
		} else {
			log.Errorf("Failed to process transaction %v: %v",
				txHash, err)
		}

		// Convert the error into an appropriate reject message and
		// send it.
		code, reason := mempool.ErrToRejectErr(err)
		peer.PushRejectMsg(wire.CmdTx, code, reason, txHash, false)
		return
	}

	sm.peerNotifier.AnnounceNewTransactions(acceptedTxs)
}

// current returns true if we believe we are synced with our peers, false if we
// still have blocks to check
func (sm *SyncManager) current() bool {
	if !sm.chain.IsCurrent() {
		return false
	}

	// if blockChain thinks we are current and we have no syncPeer it
	// is probably right.
	if sm.syncPeer == nil {
		return true
	}

	// No matter what chain thinks, if we are below the block we are syncing
	// to we are not current.
	if sm.chain.BestSnapshot().Height < sm.syncPeer.LastBlock() {
		return false
	}
	return true
}

// handleBlockMsg handles block messages from all peers.
// Returns error if Block violates consensus rules.
func (sm *SyncManager) handleBlockMsg(bmsg *blockMsg) error {
	peer := bmsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received block message from unknown peer %s", peer)
		return nil
	}

	// REDOWNLOAD fast-path: a block whose hash was approved by the peer's
	// active REDOWNLOAD session flows through the Tier-2 acceptance pipeline.
	if handled, shouldPunish, err := sm.handleRedownloadBlockArrival(
		peer, state, bmsg.block); handled {
		if err != nil && shouldPunish {
			peer.Disconnect()
		}
		return err
	}

	// If we didn't ask for this block then the peer is misbehaving.
	blockHash := bmsg.block.Hash()
	if _, exists = state.requestedBlocks[*blockHash]; !exists {
		// The regression test intentionally sends some blocks twice
		// to test duplicate block insertion fails.  Don't disconnect
		// the peer or ignore the block when we're in regression test
		// mode in this case so the chain code is actually fed the
		// duplicate blocks.
		if sm.chainParams != &chaincfg.RegressionNetParams {
			log.Warnf("Got unrequested block %v from %s -- "+
				"disconnecting", blockHash, peer.Addr())
			peer.Disconnect()
			return nil
		}
	}

	// Verify the block's parent is known and the chain has sufficient work.
	// Blocks that were explicitly requested (via fetchHeaderBlocks) already
	// passed anti-DoS validation at the header level, so skip the threshold
	// check for them.
	wasRequested := exists
	parentHash := &bmsg.block.MsgBlock().BlockHeader().PrevBlock
	parentInfo := sm.chain.LookupChainStartInfo(parentHash)
	if parentInfo == nil {
		log.Debugf("Ignoring block %v from %s with unknown parent %v",
			blockHash, peer.Addr(), parentHash)
		return nil
	}

	if !wasRequested {
		threshold := sm.getAntiDoSWorkThreshold()
		if parentInfo.WorkSum.Cmp(threshold) < 0 {
			log.Debugf("Block %v from %s: parent chain below anti-DoS "+
				"threshold, discarding block and starting presync",
				blockHash, peer.Addr())
			if state.headersSyncState == nil {
				locator := sm.chain.BlockLocatorFromHash(&parentInfo.Hash)
				state.headersSyncState = NewHeadersSyncState(
					peer.ID(),
					sm.chainParams,
					chainStartInfo{
						ChainStartInfo: *parentInfo,
						locator:        locator,
					},
					threshold,
				)
				hssLocator := state.headersSyncState.NextHeadersRequestLocator()
				state.clearHeadersRateLimit()
				sm.maybeSendGetHeaders(peer, state,
					hssLocator, &zeroHash, false)
			}
			return nil
		}
	}

	// In headers-first mode, if the block matches the first entry in the
	// header list, it's eligible for fast-add. Remove list entries except
	// for checkpoint blocks.
	isCheckpointBlock := false
	behaviorFlags := blockchain.BFNone
	if sm.headersFirstMode {
		firstNodeEl := sm.headerList.Front()
		if firstNodeEl != nil {
			firstNode := firstNodeEl.Value.(*headerNode)
			if blockHash.IsEqual(firstNode.hash) {
				behaviorFlags |= blockchain.BFFastAdd
				if sm.nextCheckpoint != nil &&
					firstNode.hash.IsEqual(sm.nextCheckpoint.Hash) {
					isCheckpointBlock = true
				} else {
					sm.headerList.Remove(firstNodeEl)
				}
			}
		}
	}

	// Remove block from request maps. Either chain will know about it and
	// so we shouldn't have any more instances of trying to fetch it, or we
	// will fail the insert and thus we'll retry next time we get an inv.
	delete(state.requestedBlocks, *blockHash)
	delete(sm.requestedBlocks, *blockHash)

	// Process the block to include validation, best chain selection, orphan
	// handling, etc.
	_, isOrphan, err := sm.chain.ProcessBlock(bmsg.block, behaviorFlags)
	if err != nil {
		// When the error is a rule error, it means the block was simply
		// rejected as opposed to something actually going wrong, so log
		// it as such.  Otherwise, something really did go wrong, so log
		// it as an actual error.
		if _, ok := err.(blockchain.RuleError); ok {
			log.Infof("Rejected block %v from %s: %v", blockHash,
				peer, err)
		} else {
			log.Errorf("Failed to process block %v: %v",
				blockHash, err)
		}
		if dbErr, ok := err.(database.Error); ok && dbErr.ErrorCode ==
			database.ErrCorruption {
			panic(dbErr)
		}

		// Convert the error into an appropriate reject message and
		// send it.
		code, reason := mempool.ErrToRejectErr(err)
		peer.PushRejectMsg(wire.CmdBlock, code, reason, blockHash, false)
		return err
	}

	if isOrphan {
		log.Debugf("Ignoring orphan block %v from %s", blockHash,
			peer.Addr())
		return nil
	}

	if peer == sm.syncPeer {
		sm.lastProgressTime = time.Now()
	}

	sm.progressLogger.LogBlockHeight(bmsg.block, sm.chain)

	best := sm.chain.BestSnapshot()

	// Update peer quality counter based on block proximity to tip.
	blockCSI := sm.chain.LookupChainStartInfo(blockHash)
	if blockCSI != nil {
		if sm.shouldDownloadBlocks(blockCSI) {
			if best.Hash == *blockHash {
				state.peerQualityCounter = 0
			}
		} else {
			state.peerQualityCounter++
		}
	}

	// Clear the rejected transactions.
	sm.rejectedTxns = make(map[chainhash.Hash]struct{})

	// Update the block height for this peer. Only relay the height update
	// to the server if we believe we're close to current, to avoid spammy
	// messages during initial sync.
	peer.UpdateLastBlockHeight(best.Height)
	if sm.current() {
		go sm.peerNotifier.UpdatePeerHeights(&best.Hash, best.Height,
			peer)
	}

	// If we are not in headers first mode, it's a good time to periodically
	// flush the blockchain cache because we don't expect new blocks immediately.
	// After that, there is nothing more to do.
	if !sm.headersFirstMode {
		if err := sm.chain.FlushUtxoCache(blockchain.FlushPeriodic); err != nil {
			log.Errorf("Error while flushing the blockchain cache: %v", err)
		}
		return nil
	}

	// This is headers-first mode, so if the block is not a checkpoint
	// request more blocks using the header list when the request queue is
	// getting short.
	if !isCheckpointBlock {
		if sm.startHeader != nil &&
			len(state.requestedBlocks) < minInFlightBlocks {
			sm.fetchMissingBlocks(&best.Hash)
		}
		return nil
	}

	// This is headers-first mode and the block is a checkpoint.  When
	// there is a next checkpoint, get the next round of headers by asking
	// for headers starting from the block after this one up to the next
	// checkpoint.
	if sm.nextCheckpoint == nil {
		return nil
	}
	prevHeight := sm.nextCheckpoint.Height
	prevHash := sm.nextCheckpoint.Hash
	sm.nextCheckpoint = sm.findNextHeaderCheckpoint(prevHeight)
	if sm.nextCheckpoint != nil {
		locator := blockchain.BlockLocator([]*chainhash.Hash{prevHash})
		err := peer.PushGetHeadersMsg(locator, sm.nextCheckpoint.Hash, true)
		if err != nil {
			log.Warnf("Failed to send getheaders message to "+
				"peer %s: %v", peer.Addr(), err)
			return nil
		}
		log.Infof("Downloading headers for blocks %d to %d from "+
			"peer %s", prevHeight+1, sm.nextCheckpoint.Height,
			sm.syncPeer.Addr())
		return nil
	}

	// Past the final checkpoint -- continue with headers-first mode, just
	// request more headers with no stop hash.
	log.Infof("Reached the final checkpoint -- continuing headers sync")
	locator := blockchain.BlockLocator([]*chainhash.Hash{blockHash})
	err = peer.PushGetHeadersMsg(locator, &zeroHash, true)
	if err != nil {
		log.Warnf("Failed to send getheaders message to peer %s: %v",
			peer.Addr(), err)
		return nil
	}
	return nil
}

// fetchMissingBlocks creates and sends a request to the syncPeer for the next
// list of blocks to be downloaded based on missing blocks in the block index.
func (sm *SyncManager) fetchMissingBlocks(tipHash *chainhash.Hash) {
	// Find the missing blocks from the index.
	missingHashes := sm.chain.LocateMissingBlockHashes(tipHash)
	if len(missingHashes) == 0 {
		return
	}

	syncPeerState := sm.peerStates[sm.syncPeer]
	if syncPeerState == nil {
		return
	}

	// Build up a getdata request for the list of missing blocks.
	// The size hint will be limited to wire.MaxInvPerMsg by the function.
	gdmsg := wire.NewMsgGetDataSizeHint(uint(len(missingHashes)))
	numRequested := 0

	for _, hash := range missingHashes {
		if _, exists := syncPeerState.requestedBlocks[*hash]; exists {
			continue
		}

		iv := wire.NewInvVect(wire.InvTypeWitnessBlock, hash)
		sm.requestedBlocks[*hash] = struct{}{}
		syncPeerState.requestedBlocks[*hash] = struct{}{}

		gdmsg.AddInvVect(iv)
		numRequested++

		if numRequested >= wire.MaxInvPerMsg {
			break
		}
	}

	if len(gdmsg.InvList) > 0 {
		sm.syncPeer.QueueMessage(gdmsg, nil)
	}
}

// handleHeadersMsg is the universal entry point for all incoming headers.
// It accepts headers from any peer, routes through presync for low-work
// chains, and stores headers via AcceptBlockHeader for sufficient-work chains.
// Returns a non-nil error when the peer misbehaved.
func (sm *SyncManager) handleHeadersMsg(hmsg *headersMsg) error {
	peer := hmsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received headers message from unknown peer %s", peer)
		return nil
	}

	headers := hmsg.headers.Headers
	numHeaders := len(headers)

	// Empty message: abandon any active presync and mark the peer low-quality.
	// Clear the rate limit so a follow-up can go out immediately.
	if numHeaders == 0 {
		state.clearHeadersRateLimit()
		if state.headersSyncState != nil {
			state.peerQualityCounter = peerQualityThreshold + 1
			state.headersSyncState = nil
			state.pipelinedLocatorHead = nil
			log.Debugf("Presync abandoned: peer %s sent empty headers",
				peer.Addr())
		}
		return nil
	}

	// Pre-validation: nBits format and intra-batch continuity.
	if err := sm.preValidateHeaders(peer, headers); err != nil {
		return err
	}

	// Inconsistent certificate inclusion within a single HEADERS message
	// (some headers have certs, others don't) is a protocol violation.
	if hasInconsistentCerts(headers) {
		state.headersSyncState = nil
		return ErrInconsistentCerts
	}

	// PIPELINING STALE-DROP GUARD
	//
	// If an outstanding speculative GETHEADERS has locator tip matching this
	// batch's prev-hash, check whether live post-processing state still expects
	// that tip. If not (phase transitioned, session aborted, etc.), silently
	// drop the response without disconnecting or penalising the peer.
	if state.pipelinedLocatorHead != nil &&
		headers[0].BlockHeader.PrevBlock == *state.pipelinedLocatorHead {

		prevBlock := headers[0].BlockHeader.PrevBlock
		state.pipelinedLocatorHead = nil
		if pipelinedSpeculativeGetHeadersIsStale(
			state.headersSyncState, prevBlock) {
			log.Debugf("Pipelined getheaders: dropping speculative "+
				"HEADERS from peer %s (live state diverged)",
				peer.Addr())
			return nil
		}
		// Live state matches: fall through to normal processing.
	}

	// SPECULATIVE DISPATCH
	//
	// When the batch structurally continues the current phase tip and no
	// speculation is already in flight, dispatch the next GETHEADERS now so
	// the peer starts producing the next batch in parallel with local ZK
	// verification of the current batch.
	var speculatedTip *chainhash.Hash
	if hss := state.headersSyncState; hss != nil &&
		numHeaders == wire.MaxBlockHeadersPerMsg &&
		state.pipelinedLocatorHead == nil {

		var phaseTipMatches bool
		switch hss.Phase() {
		case PhasePresync:
			phaseTipMatches = headers[0].BlockHeader.PrevBlock ==
				hss.LastHeaderHash() && !hss.SpotCheckBackpressured()
		case PhaseRedownload:
			phaseTipMatches = headers[0].BlockHeader.PrevBlock ==
				hss.RedownloadTipHash()
		}
		if phaseTipMatches {
			specTip := headers[numHeaders-1].BlockHeader.BlockHash()
			locator := hss.SpeculativeLocator(specTip)
			// Clear rate limit: a full phase-continuing batch is a
			// plausible response to our outstanding getheaders.
			state.clearHeadersRateLimit()
			if sm.maybeSendGetHeaders(peer, state, locator,
				&zeroHash, false) {
				specTipCopy := specTip
				state.pipelinedLocatorHead = &specTipCopy
				speculatedTip = &specTipCopy
			}
		}
	}

	// Route through an active presync session.
	if state.headersSyncState != nil {
		res, err := sm.isContinuationOfLowWorkHeadersSync(
			peer, state, headers, speculatedTip)
		if err != nil {
			return err
		}
		if res.handled {
			sm.driveRedownloadGetdata(peer, state, res.newlyApproved)
			if res.justFinalized {
				return sm.acceptValidatedHeaders(peer, state, headers[:0], true)
			}
			return nil
		}
	}

	// Find the fork point: the first header's parent in our block index.
	firstPrevHash := &headers[0].BlockHeader.PrevBlock
	chainStart := sm.chain.LookupChainStartInfo(firstPrevHash)
	if chainStart == nil {
		// Headers don't connect to anything we know. Ask for more
		// headers from our best header to fill the gap. Do NOT clear the
		// rate limit: disconnected batches are not plausible getheaders responses.
		locator, err := sm.chain.LatestBlockLocator()
		if err != nil {
			log.Warnf("Failed to get block locator: %v", err)
			return nil
		}
		sm.maybeSendGetHeaders(peer, state, locator, &zeroHash, true)
		return nil
	}

	// Headers connect to something we know: treat as a plausible response.
	state.clearHeadersRateLimit()

	claimedWork := calculateClaimedHeadersWork(headers)
	totalWork := new(big.Int).Add(chainStart.WorkSum, claimedWork)
	threshold := sm.getAntiDoSWorkThreshold()

	if totalWork.Cmp(threshold) < 0 {
		// Low work: attempt presync if the batch is full.
		return sm.tryLowWorkHeadersSync(
			peer, state, chainStart, headers, threshold)
	}

	// Sufficient work: certificates are required to accept directly.
	if headers[0].BlockCertificate() == nil {
		locator := blockchain.BlockLocator([]*chainhash.Hash{firstPrevHash})
		sm.maybeSendGetHeaders(peer, state, locator, &zeroHash, true)
		return nil
	}

	shouldProbe := numHeaders == wire.MaxBlockHeadersPerMsg
	return sm.acceptValidatedHeaders(peer, state, headers, shouldProbe)
}

// hasInconsistentCerts returns true when a HEADERS message contains a mix of
// headers with and without certificates. This is a protocol violation.
func hasInconsistentCerts(headers []wire.MsgHeader) bool {
	hasCert, noCert := false, false
	for i := range headers {
		if headers[i].BlockCertificate() != nil {
			hasCert = true
		} else {
			noCert = true
		}
		if hasCert && noCert {
			return true
		}
	}
	return false
}

// preValidateHeaders checks nBits format validity and intra-batch continuity.
// Returns an error wrapping ErrPeerViolation on failure.
func (sm *SyncManager) preValidateHeaders(peer *peerpkg.Peer,
	headers []wire.MsgHeader) error {

	powLimit := sm.chainParams.PowLimit
	var prevHash chainhash.Hash

	for i := range headers {
		bh := &headers[i].BlockHeader

		// nBits must encode a valid target within powLimit.
		target := blockchain.CompactToBig(bh.Bits)
		if target.Sign() <= 0 || target.Cmp(powLimit) > 0 {
			return blockchain.RuleError{
				ErrorCode: blockchain.ErrUnexpectedDifficulty,
				Description: fmt.Sprintf("peer %s sent header with invalid "+
					"nBits 0x%08x", peer.Addr(), bh.Bits),
			}
		}

		// Intra-batch continuity: each header's prevHash must match
		// the preceding header's hash (except the first).
		if i > 0 && bh.PrevBlock != prevHash {
			return fmt.Errorf("peer %s sent non-continuous headers "+
				"at index %d: %w", peer.Addr(), i,
				ErrPeerViolation)
		}
		prevHash = bh.BlockHash()
	}
	return nil
}

// acceptHeadersIntoIndex validates and stores headers via AcceptBlockHeader,
// then enqueues newly accepted ones for block fetching.
// Returns the last accepted header (or nil if none were accepted) and an error
// if any header was rejected.
func (sm *SyncManager) acceptHeadersIntoIndex(
	hdrs []wire.MsgHeader,
) (*blockchain.AcceptedHeader, []*blockchain.AcceptedHeader, error) {

	var lastAccepted *blockchain.AcceptedHeader
	var acceptedHeaders []*blockchain.AcceptedHeader

	for i := range hdrs {
		ah, isNew, err := sm.chain.AcceptBlockHeader(
			&hdrs[i].BlockHeader, hdrs[i].BlockCertificate())
		if err != nil {
			return lastAccepted, acceptedHeaders, err
		}
		if isNew {
			acceptedHeaders = append(acceptedHeaders, ah)
		}
		lastAccepted = ah
	}
	return lastAccepted, acceptedHeaders, nil
}

// enqueueHeadersForFetch adds accepted headers to the header list used by
// the checkpoint-based headers-first mode.
func (sm *SyncManager) enqueueHeadersForFetch(
	acceptedHeaders []*blockchain.AcceptedHeader,
) {
	if !sm.headersFirstMode {
		return
	}
	for _, ah := range acceptedHeaders {
		hash := ah.Hash
		node := &headerNode{hash: &hash, height: ah.Height}
		e := sm.headerList.PushBack(node)
		if sm.startHeader == nil {
			sm.startHeader = e
		}
	}
}

// acceptValidatedHeaders stores cert-validated headers into the block index,
// advances per-peer tracking, schedules block downloads, and probes the peer
// for more headers when appropriate.
//
// shouldProbe is set when the wire batch was full or the presync state machine
// just finalised (justFinalized), signalling the peer may have more headers.
// The probe is suppressed while an active presync session is driving its own
// getheaders, and when the peer is already at or behind our locator head.
func (sm *SyncManager) acceptValidatedHeaders(
	peer *peerpkg.Peer, state *peerSyncState,
	headers []wire.MsgHeader, shouldProbe bool,
) error {
	if len(headers) == 0 && !shouldProbe {
		return nil
	}

	var lastAccepted *blockchain.AcceptedHeader
	var acceptedHeaders []*blockchain.AcceptedHeader
	var err error
	if len(headers) > 0 {
		lastAccepted, acceptedHeaders, err = sm.acceptHeadersIntoIndex(headers)
		if err != nil {
			log.Warnf("Failed to accept block header from peer %s: %v",
				peer.Addr(), err)
			return err
		}
	}

	var locatorHead chainhash.Hash
	var peerHeight int32
	if lastAccepted != nil {
		locatorHead = lastAccepted.Hash
		peerHeight = lastAccepted.Height
		peer.UpdateLastBlockHeight(peerHeight)
		sm.enqueueHeadersForFetch(acceptedHeaders)
		csi := sm.chain.LookupChainStartInfo(&locatorHead)
		if csi != nil && sm.shouldDownloadBlocks(csi) {
			sm.fetchMissingBlocks(&locatorHead)
		} else if csi != nil && len(acceptedHeaders) > 0 {
			state.peerQualityCounter++
		}
	} else if len(headers) > 0 {
		locatorHead = headers[len(headers)-1].BlockHeader.BlockHash()
		peerHeight = peer.LastBlock()
	}

	if shouldProbe && state.headersSyncState == nil &&
		peer.LastBlock() > peerHeight && !locatorHead.IsEqual(&zeroHash) {
		locator := blockchain.BlockLocator([]*chainhash.Hash{&locatorHead})
		sm.maybeSendGetHeaders(peer, state, locator, &zeroHash, true)
	}
	return nil
}

// handleNotFoundMsg handles notfound messages from all peers.
func (sm *SyncManager) handleNotFoundMsg(nfmsg *notFoundMsg) {
	peer := nfmsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received notfound message from unknown peer %s", peer)
		return
	}
	for _, inv := range nfmsg.notFound.InvList {
		// verify the hash was actually announced by the peer
		// before deleting from the global requested maps.
		switch inv.Type {
		case wire.InvTypeWitnessBlock:
			fallthrough
		case wire.InvTypeBlock:
			if _, exists := state.requestedBlocks[inv.Hash]; exists {
				delete(state.requestedBlocks, inv.Hash)
				delete(sm.requestedBlocks, inv.Hash)
			}

		case wire.InvTypeWitnessTx:
			fallthrough
		case wire.InvTypeTx:
			if _, exists := state.requestedTxns[inv.Hash]; exists {
				delete(state.requestedTxns, inv.Hash)
				delete(sm.requestedTxns, inv.Hash)
			}
		}
	}
}

// haveInventory returns whether or not the inventory represented by the passed
// inventory vector is known.  This includes checking all of the various places
// inventory can be when it is in different states such as blocks that are part
// of the main chain, on a side chain, in the orphan pool, and transactions that
// are in the memory pool (either the main pool or orphan pool).
func (sm *SyncManager) haveInventory(invVect *wire.InvVect) (bool, error) {
	switch invVect.Type {
	case wire.InvTypeWitnessBlock:
		fallthrough
	case wire.InvTypeBlock:
		// Only consider a block "known" if we have the full block data,
		// not just a header-only entry from AcceptBlockHeader.
		if sm.chain.HaveBlockData(&invVect.Hash) {
			return true, nil
		}
		return sm.chain.IsKnownOrphan(&invVect.Hash), nil

	case wire.InvTypeWitnessTx:
		fallthrough
	case wire.InvTypeTx:
		// Ask the transaction memory pool if the transaction is known
		// to it in any form (main pool or orphan).
		if sm.txMemPool.HaveTransaction(&invVect.Hash) {
			return true, nil
		}

		// Check if the transaction exists from the point of view of the
		// end of the main chain.  Note that this is only a best effort
		// since it is expensive to check existence of every output and
		// the only purpose of this check is to avoid downloading
		// already known transactions.  Only the first two outputs are
		// checked because the vast majority of transactions consist of
		// two outputs where one is some form of "pay-to-somebody-else"
		// and the other is a change output.
		prevOut := wire.OutPoint{Hash: invVect.Hash}
		for i := uint32(0); i < 2; i++ {
			prevOut.Index = i
			entry, err := sm.chain.FetchUtxoEntry(prevOut)
			if err != nil {
				return false, err
			}
			if entry != nil && !entry.IsSpent() {
				return true, nil
			}
		}

		return false, nil
	}

	// The requested inventory is an unsupported type, so just claim
	// it is known to avoid requesting it.
	return true, nil
}

// handleInvMsg handles inv messages from all peers.
// We examine the inventory advertised by the remote peer and act accordingly.
func (sm *SyncManager) handleInvMsg(imsg *invMsg) {
	peer := imsg.peer
	state, exists := sm.peerStates[peer]
	if !exists {
		log.Warnf("Received inv message from unknown peer %s", peer)
		return
	}

	// Attempt to find the final block in the inventory list.  There may
	// not be one.
	lastBlock := -1
	invVects := imsg.inv.InvList
	for i := len(invVects) - 1; i >= 0; i-- {
		if invVects[i].Type == wire.InvTypeBlock {
			lastBlock = i
			break
		}
	}

	// If this inv contains a block announcement, and this isn't coming from
	// our current sync peer or we're current, then update the last
	// announced block for this peer. We'll use this information later to
	// update the heights of peers based on blocks we've accepted that they
	// previously announced.
	if lastBlock != -1 && (peer != sm.syncPeer || sm.current()) {
		peer.UpdateLastAnnouncedBlock(&invVects[lastBlock].Hash)
	}

	// Ignore invs from peers that aren't the sync if we are not current.
	// Helps prevent fetching a mass of orphans.
	if peer != sm.syncPeer && !sm.current() {
		return
	}

	// If our chain is current and a peer announces a block we already
	// know of, then update their current block height.
	if lastBlock != -1 && sm.current() {
		blkHeight, err := sm.chain.BlockHeightByHash(&invVects[lastBlock].Hash)
		if err == nil {
			peer.UpdateLastBlockHeight(blkHeight)
		}
	}

	// Request the advertised inventory if we don't already have it.  Also,
	// request parent blocks of orphans if we receive one we already have.
	// Finally, attempt to detect potential stalls due to long side chains
	// we already have and request more blocks to prevent them.
	sentGetHeaders := false
	for i, iv := range invVects {
		// Ignore unsupported inventory types.
		switch iv.Type {
		case wire.InvTypeBlock:
		case wire.InvTypeTx:
		case wire.InvTypeWitnessBlock:
		case wire.InvTypeWitnessTx:
		default:
			continue
		}

		// Add the inventory to the cache of known inventory
		// for the peer.
		peer.AddKnownInventory(iv)

		// During initial sync (not current), skip inv processing to
		// avoid requesting blocks before headers are indexed.
		if sm.headersFirstMode && !sm.current() {
			continue
		}

		// Request the inventory if we don't already have it.
		haveInv, err := sm.haveInventory(iv)
		if err != nil {
			log.Warnf("Unexpected failure when checking for "+
				"existing inventory during inv message "+
				"processing: %v", err)
			continue
		}
		if !haveInv {
			if iv.Type == wire.InvTypeTx {
				// Skip the transaction if it has already been
				// rejected.
				if _, exists := sm.rejectedTxns[iv.Hash]; exists {
					continue
				}
			}

			// For unknown blocks from low-quality peers, request
			// headers first (inv → headers → getdata) instead of
			// fetching the block directly.
			if iv.Type == wire.InvTypeBlock && sm.current() &&
				!isPeerHighQuality(state) {

				if !sentGetHeaders {
					locator, err := sm.chain.LatestBlockLocator()
					if err == nil {
						_ = sm.pushGetHeadersDirect(peer, state, locator, &zeroHash, true)
					}
					sentGetHeaders = true
				}
				continue
			}

			// Add it to the request queue for getdata.
			state.requestQueue = append(state.requestQueue, iv)
			continue
		}

		if iv.Type == wire.InvTypeBlock {
			// We already know this block. If it's the final
			// advertised block and is not our tip, request
			// headers from it to discover any chain extension.
			if i == lastBlock {
				best := sm.chain.BestSnapshot()
				if !iv.Hash.IsEqual(&best.Hash) {
					locator := sm.chain.BlockLocatorFromHash(&iv.Hash)
					_ = sm.pushGetHeadersDirect(peer, state, locator, &zeroHash, true)
				}
			}
		}
	}

	// Request as much as possible at once.  Anything that won't fit into
	// the request will be requested on the next inv message.
	numRequested := 0
	gdmsg := wire.NewMsgGetData()
	requestQueue := state.requestQueue
	for len(requestQueue) != 0 {
		iv := requestQueue[0]
		requestQueue[0] = nil
		requestQueue = requestQueue[1:]

		switch iv.Type {
		case wire.InvTypeWitnessBlock:
			fallthrough
		case wire.InvTypeBlock:
			// Request the block if there is not already a pending
			// request.
			if _, exists := sm.requestedBlocks[iv.Hash]; !exists {
				limitAdd(sm.requestedBlocks, iv.Hash, maxRequestedBlocks)
				limitAdd(state.requestedBlocks, iv.Hash, maxRequestedBlocks)

				iv.Type = wire.InvTypeWitnessBlock
				gdmsg.AddInvVect(iv)
				numRequested++
			}

		case wire.InvTypeWitnessTx:
			fallthrough
		case wire.InvTypeTx:
			// Request the transaction if there is not already a
			// pending request.
			if _, exists := sm.requestedTxns[iv.Hash]; !exists {
				limitAdd(sm.requestedTxns, iv.Hash, maxRequestedTxns)
				limitAdd(state.requestedTxns, iv.Hash, maxRequestedTxns)

				iv.Type = wire.InvTypeWitnessTx
				gdmsg.AddInvVect(iv)
				numRequested++
			}
		}

		if numRequested >= wire.MaxInvPerMsg {
			break
		}
	}
	state.requestQueue = requestQueue
	if len(gdmsg.InvList) > 0 {
		peer.QueueMessage(gdmsg, nil)
	}
}

// processMessage handles a single message from the block handler queue.
func (sm *SyncManager) processMessage(m interface{}) {
	switch msg := m.(type) {
	case *newPeerMsg:
		sm.handleNewPeerMsg(msg.peer)

	case *txMsg:
		sm.handleTxMsg(msg)
		msg.reply <- struct{}{}

	case *blockMsg:
		msg.reply <- sm.handleBlockMsg(msg)

	case *invMsg:
		sm.handleInvMsg(msg)

	case *headersMsg:
		if err := sm.handleHeadersMsg(msg); err != nil {
			sm.peerVerdicts <- PeerVerdict{PeerID: msg.peer.ID(), Err: err}
		}

	case *notFoundMsg:
		sm.handleNotFoundMsg(msg)

	case *donePeerMsg:
		sm.handleDonePeerMsg(msg.peer)

	case getSyncPeerMsg:
		var peerID int32
		if sm.syncPeer != nil {
			peerID = sm.syncPeer.ID()
		}
		msg.reply <- peerID

	case processBlockMsg:
		_, isOrphan, err := sm.chain.ProcessBlock(
			msg.block, msg.flags)
		if err != nil {
			msg.reply <- processBlockResponse{
				isOrphan: false,
				err:      err,
			}
		}

		msg.reply <- processBlockResponse{
			isOrphan: isOrphan,
			err:      nil,
		}

	case isCurrentMsg:
		msg.reply <- sm.current()

	case pauseMsg:
		// Wait until the sender unpauses the manager.
		<-msg.unpause

	default:
		log.Warnf("Invalid message type in block "+
			"handler: %T", msg)
	}
}

// blockHandler is the main handler for the sync manager.  It must be run as a
// goroutine.  It processes block and inv messages in a separate goroutine
// from the peer handlers so the block (MsgBlock) messages are handled by a
// single thread without needing to lock memory data structures.  This is
// important because the sync manager controls which blocks are needed and how
// the fetching should proceed.
func (sm *SyncManager) blockHandler() {
	stallTicker := time.NewTicker(stallSampleInterval)
	defer stallTicker.Stop()

	maxQueueSize := cap(sm.msgChan)
	queue := list.New()

	// How often (in messages processed) to check for a high-priority block to process out of order.
	oooPeriod := 2 + maxQueueSize/30
	counter := 0

out:
	for {
		// If the queue is empty, block until a message arrives.
		if queue.Len() == 0 {
			counter = 0
			select {
			case m := <-sm.msgChan:
				queue.PushBack(m)
			case <-stallTicker.C:
				sm.handleStallSample()
				continue
			case <-sm.quit:
				break out
			}
		}

		// Drain pending messages from the channel into the queue.
	drain:
		for queue.Len() < maxQueueSize {
			select {
			case m := <-sm.msgChan:
				queue.PushBack(m)
			default:
				break drain
			}
		}

		// Every tipPeriod messages, check for a priority block.
		elem := queue.Front()
		if counter == 0 {
			elem = findBestBlockMsg(sm.chain, queue)
		}
		counter = (counter + 1) % oooPeriod

		msg := queue.Remove(elem)

		sm.processMessage(msg)

		select {
		case <-stallTicker.C:
			sm.handleStallSample()
		case <-sm.quit:
			break out
		default:
		}
	}

	log.Debug("Block handler shutting down: flushing blockchain caches...")
	if err := sm.chain.FlushUtxoCache(blockchain.FlushRequired); err != nil {
		log.Errorf("Error while flushing blockchain caches: %v", err)
	}

	sm.wg.Done()
	log.Trace("Block handler done")
}

// handleBlockchainNotification handles notifications from blockchain.  It does
// things such as request orphan block parents and relay accepted blocks to
// connected peers.
func (sm *SyncManager) handleBlockchainNotification(notification *blockchain.Notification) {
	switch notification.Type {
	// A block has been accepted into the block chain.  Relay it to other
	// peers.
	case blockchain.NTBlockAccepted:
		// Don't relay if we are not current. Other peers that are
		// current should already know about it.
		if !sm.current() {
			return
		}

		block, ok := notification.Data.(*btcutil.Block)
		if !ok {
			log.Warnf("Chain accepted notification is not a block.")
			break
		}

		// Generate the inventory vector and relay it.
		iv := wire.NewInvVect(wire.InvTypeBlock, block.Hash())
		sm.peerNotifier.RelayInventory(iv, block.MsgBlock().MsgHeader)

	// A block has been connected to the main block chain.
	case blockchain.NTBlockConnected:
		// Don't attempt to update the mempool if we're not current.
		// The mempool is empty and the fee estimator is useless unless
		// we're caught up.
		if !sm.current() {
			return
		}

		block, ok := notification.Data.(*btcutil.Block)
		if !ok {
			log.Warnf("Chain connected notification is not a block.")
			break
		}

		// Remove all of the transactions (except the coinbase) in the
		// connected block from the transaction pool.  Secondly, remove any
		// transactions which are now double spends as a result of these
		// new transactions.  Finally, remove any transaction that is
		// no longer an orphan. Transactions which depend on a confirmed
		// transaction are NOT removed recursively because they are still
		// valid.
		for _, tx := range block.Transactions()[1:] {
			sm.txMemPool.RemoveTransaction(tx, false)
			sm.txMemPool.RemoveDoubleSpends(tx)
			sm.txMemPool.RemoveOrphan(tx)
			sm.peerNotifier.TransactionConfirmed(tx)
			acceptedTxs := sm.txMemPool.ProcessOrphans(tx)
			sm.peerNotifier.AnnounceNewTransactions(acceptedTxs)
		}

		// Register block with the fee estimator, if it exists.
		if sm.feeEstimator != nil {
			err := sm.feeEstimator.RegisterBlock(block)

			// If an error is somehow generated then the fee estimator
			// has entered an invalid state. Since it doesn't know how
			// to recover, create a new one.
			if err != nil {
				sm.feeEstimator = mempool.NewFeeEstimator(
					mempool.DefaultEstimateFeeMaxRollback,
					mempool.DefaultEstimateFeeMinRegisteredBlocks)
			}
		}

	// A block has been disconnected from the main block chain.
	case blockchain.NTBlockDisconnected:
		block, ok := notification.Data.(*btcutil.Block)
		if !ok {
			log.Warnf("Chain disconnected notification is not a block.")
			break
		}

		// Reinsert all of the transactions (except the coinbase) into
		// the transaction pool.
		for _, tx := range block.Transactions()[1:] {
			_, _, err := sm.txMemPool.MaybeAcceptTransaction(tx,
				false, false)
			if err != nil {
				// Remove the transaction and all transactions
				// that depend on it if it wasn't accepted into
				// the transaction pool.
				sm.txMemPool.RemoveTransaction(tx, true)
			}
		}

		// Rollback previous block recorded by the fee estimator.
		if sm.feeEstimator != nil {
			sm.feeEstimator.Rollback(block.Hash())
		}
	}
}

// NewPeer informs the sync manager of a newly active peer.
func (sm *SyncManager) NewPeer(peer *peerpkg.Peer) {
	// Ignore if we are shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}
	sm.msgChan <- &newPeerMsg{peer: peer}
}

// QueueTx adds the passed transaction message and peer to the block handling
// queue. Responds to the done channel argument after the tx message is
// processed.
func (sm *SyncManager) QueueTx(tx *btcutil.Tx, peer *peerpkg.Peer, done chan struct{}) {
	// Don't accept more transactions if we're shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		done <- struct{}{}
		return
	}

	sm.msgChan <- &txMsg{tx: tx, peer: peer, reply: done}
}

// QueueBlock adds the passed block message and peer to the block handling
// queue. Responds to the done channel argument after the block message is
// processed.
func (sm *SyncManager) QueueBlock(block *btcutil.Block, peer *peerpkg.Peer, done chan error) {
	// Don't accept more blocks if we're shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		done <- nil
		return
	}

	sm.msgChan <- &blockMsg{block: block, peer: peer, reply: done}
}

// QueueInv adds the passed inv message and peer to the block handling queue.
func (sm *SyncManager) QueueInv(inv *wire.MsgInv, peer *peerpkg.Peer) {
	// No channel handling here because peers do not need to block on inv
	// messages.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	sm.msgChan <- &invMsg{inv: inv, peer: peer}
}

// QueueHeaders adds the passed headers message and peer to the block handling
// queue.
func (sm *SyncManager) QueueHeaders(headers *wire.MsgHeaders, peer *peerpkg.Peer) {
	// No channel handling here because peers do not need to block on
	// headers messages.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	sm.msgChan <- &headersMsg{headers: headers, peer: peer}
}

// QueueNotFound adds the passed notfound message and peer to the block handling
// queue.
func (sm *SyncManager) QueueNotFound(notFound *wire.MsgNotFound, peer *peerpkg.Peer) {
	// No channel handling here because peers do not need to block on
	// reject messages.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	sm.msgChan <- &notFoundMsg{notFound: notFound, peer: peer}
}

// DonePeer informs the blockmanager that a peer has disconnected.
func (sm *SyncManager) DonePeer(peer *peerpkg.Peer) {
	// Ignore if we are shutting down.
	if atomic.LoadInt32(&sm.shutdown) != 0 {
		return
	}

	sm.msgChan <- &donePeerMsg{peer: peer}
}

// Start begins the core block handler which processes block and inv messages.
func (sm *SyncManager) Start() {
	// Already started?
	if atomic.AddInt32(&sm.started, 1) != 1 {
		return
	}

	log.Trace("Starting sync manager")
	sm.wg.Add(1)
	go sm.blockHandler()
}

// Stop gracefully shuts down the sync manager by stopping all asynchronous
// handlers and waiting for them to finish.
func (sm *SyncManager) Stop() error {
	if atomic.AddInt32(&sm.shutdown, 1) != 1 {
		log.Warnf("Sync manager is already in the process of " +
			"shutting down")
		return nil
	}

	log.Infof("Sync manager shutting down")
	close(sm.quit)
	sm.wg.Wait()
	return nil
}

// PeerVerdicts returns a channel on which the SyncManager posts peer verdicts
// after header validation. The server layer should read from this channel to
// apply punishment (ban or disconnect) through its own policy.
func (sm *SyncManager) PeerVerdicts() <-chan PeerVerdict {
	return sm.peerVerdicts
}

// SyncPeerID returns the ID of the current sync peer, or 0 if there is none.
func (sm *SyncManager) SyncPeerID() int32 {
	reply := make(chan int32)
	sm.msgChan <- getSyncPeerMsg{reply: reply}
	return <-reply
}

// ProcessBlock makes use of ProcessBlock on an internal instance of a block
// chain.
func (sm *SyncManager) ProcessBlock(block *btcutil.Block, flags blockchain.BehaviorFlags) (bool, error) {
	reply := make(chan processBlockResponse, 1)
	sm.msgChan <- processBlockMsg{block: block, flags: flags, reply: reply}
	response := <-reply
	return response.isOrphan, response.err
}

// IsCurrent returns whether or not the sync manager believes it is synced with
// the connected peers.
func (sm *SyncManager) IsCurrent() bool {
	reply := make(chan bool)
	sm.msgChan <- isCurrentMsg{reply: reply}
	return <-reply
}

// Pause pauses the sync manager until the returned channel is closed.
//
// Note that while paused, all peer and block processing is halted.  The
// message sender should avoid pausing the sync manager for long durations.
func (sm *SyncManager) Pause() chan<- struct{} {
	c := make(chan struct{})
	sm.msgChan <- pauseMsg{c}
	return c
}

// getAntiDoSWorkThreshold returns the minimum cumulative work a chain must
// have before it is accepted without presync.
// threshold = max(tip.workSum - 100_blocks_of_tip_work, MinimumChainWork)
func (sm *SyncManager) getAntiDoSWorkThreshold() *big.Int {
	best := sm.chain.BestSnapshot()
	tipWork := blockchain.CalcWork(best.Bits)

	bufferBlocks := antiDoSBufferBlocks
	deduction := new(big.Int).Mul(tipWork, big.NewInt(bufferBlocks))

	nearTipWork := new(big.Int).Sub(best.WorkSum, deduction)
	if nearTipWork.Sign() < 0 {
		nearTipWork.SetInt64(0)
	}

	return blockchain.MaxBigInt(nearTipWork, sm.chainParams.MinimumChainWork)
}

// shouldDownloadBlocks returns true when a chain whose last known block is
// described by csi has sufficient work to merit full-block download.
//
// The criterion: csi.WorkSum + CalcWork(nextHypotheticalBits) >= bestTip.workSum
// The difficulty of the hypothetical next block is derived from the last two
// blocks of the chain (WTEMA).
func (sm *SyncManager) shouldDownloadBlocks(csi *blockchain.ChainStartInfo) bool {
	best := sm.chain.BestSnapshot()

	if csi.WorkSum.Cmp(best.WorkSum) >= 0 {
		return true
	}

	nextBits, err := blockchain.CalcNextRequiredDifficultyFromValues(
		sm.chainParams, csi.Height, csi.Bits,
		csi.Timestamp, csi.PrevTimestamp,
	)
	if err != nil {
		nextBits = sm.chainParams.PowLimitBits
	}
	nextWork := blockchain.CalcWork(nextBits)

	speculative := new(big.Int).Add(csi.WorkSum, nextWork)
	return speculative.Cmp(best.WorkSum) >= 0
}

// maybeSendGetHeaders sends a getheaders to the peer if rate-limit allows.
// Returns true if the message was sent.
func (sm *SyncManager) maybeSendGetHeaders(peer *peerpkg.Peer,
	state *peerSyncState, locator blockchain.BlockLocator,
	stopHash *chainhash.Hash, includeCerts bool) bool {

	now := time.Now()
	if now.Sub(state.lastGetHeadersTime) < headersResponseTime {
		return false
	}
	if err := peer.PushGetHeadersMsg(locator, stopHash, includeCerts); err != nil {
		log.Warnf("Failed to send getheaders to peer %s: %v",
			peer.Addr(), err)
		return false
	}
	state.lastGetHeadersTime = now
	return true
}

// calculateClaimedHeadersWork sums the work for a slice of headers.
func calculateClaimedHeadersWork(headers []wire.MsgHeader) *big.Int {
	total := new(big.Int)
	for i := range headers {
		total.Add(total, blockchain.CalcWork(headers[i].BlockHeader.Bits))
	}
	return total
}

// tryLowWorkHeadersSync creates a presync session for low-work headers from
// a peer. The caller has already verified that totalWork < threshold.
func (sm *SyncManager) tryLowWorkHeadersSync(
	peer *peerpkg.Peer, state *peerSyncState,
	chainStart *blockchain.ChainStartInfo,
	headers []wire.MsgHeader,
	threshold *big.Int,
) error {

	if len(headers) != wire.MaxBlockHeadersPerMsg {
		log.Debugf("Ignoring low-work chain (height=%d) from peer=%d",
			chainStart.Height+int32(len(headers)), peer.ID())
		return nil
	}

	locator := sm.chain.BlockLocatorFromHash(&chainStart.Hash)
	state.headersSyncState = NewHeadersSyncState(
		peer.ID(),
		sm.chainParams,
		chainStartInfo{
			ChainStartInfo: *chainStart,
			locator:        locator,
		},
		threshold,
	)
	_, err := sm.isContinuationOfLowWorkHeadersSync(peer, state, headers, nil)
	return err
}

// presyncContinuationResult describes the outcome of routing a HEADERS batch
// through the peer's active presync state machine.
type presyncContinuationResult struct {
	// newlyApproved lists REDOWNLOAD entries just pushed onto Tier-1.
	newlyApproved []ApprovedRedownloadEntry
	// handled reports whether the state machine consumed the batch.
	handled bool
	// justFinalized reports whether this call transitioned the state machine
	// to PhaseFinal, signalling the acceptance tail to probe for more headers.
	justFinalized bool
}

// isContinuationOfLowWorkHeadersSync routes headers through the peer's active
// presync state machine. Returns a presyncContinuationResult and an optional
// error when the peer misbehaved.
//
// speculatedTip is non-nil when the caller has already dispatched a speculative
// GETHEADERS. After processing, this function decides whether the speculation
// covers the RequestMore follow-up (skipping the in-function send) or whether
// the state has diverged and a recovery send is needed.
func (sm *SyncManager) isContinuationOfLowWorkHeadersSync(
	peer *peerpkg.Peer, state *peerSyncState,
	headers []wire.MsgHeader, speculatedTip *chainhash.Hash,
) (presyncContinuationResult, error) {

	hss := state.headersSyncState
	if hss == nil {
		return presyncContinuationResult{}, nil
	}

	fullMsg := len(headers) == wire.MaxBlockHeadersPerMsg
	phaseBefore := hss.Phase()
	result := hss.ProcessNextHeaders(headers, fullMsg)

	if result.ShouldPunish {
		return presyncContinuationResult{handled: true}, fmt.Errorf(
			"peer %s misbehaved during presync: %w",
			peer.Addr(), ErrPeerViolation)
	} else if !result.Success {
		state.peerQualityCounter = peerQualityThreshold + 1
	} else {
		// Valid presync continuation: clear the rate limit before the
		// in-function follow-up send. Mirrors Bitcoin Core line 2689.
		state.clearHeadersRateLimit()
	}

	if result.RequestMore {
		locator := hss.NextHeadersRequestLocator()

		skipSend := false
		if speculatedTip != nil && len(locator) > 0 &&
			*locator[0] == *speculatedTip {
			skipSend = true
		} else if speculatedTip != nil {
			state.clearHeadersRateLimit()
		}
		if !skipSend && state.pipelinedLocatorHead != nil &&
			len(locator) > 0 && *locator[0] == *state.pipelinedLocatorHead {
			skipSend = true
		}
		if !skipSend {
			sm.maybeSendGetHeaders(peer, state,
				locator, &zeroHash, false)
		}
	}

	// Send spot-check requests independently of the regular pipeline.
	for i := range result.SpotCheckRequests {
		sc := &result.SpotCheckRequests[i]
		peer.PushGetHeadersMsg(sc.Locator, &sc.StopHash, true)
	}
	if len(result.SpotCheckRequests) > 0 {
		state.lastGetHeadersTime = time.Now()
	}

	justFinalized := phaseBefore != PhaseFinal && hss.Phase() == PhaseFinal
	if hss.Phase() == PhaseFinal {
		state.headersSyncState = nil
		state.pipelinedLocatorHead = nil
	}

	return presyncContinuationResult{
		newlyApproved: result.NewlyApproved,
		handled:       true,
		justFinalized: justFinalized,
	}, nil
}

// pipelinedSpeculativeGetHeadersIsStale returns true when the speculative
// GETHEADERS response should be silently dropped because the live state no
// longer matches the expectation used to issue the speculation.
func pipelinedSpeculativeGetHeadersIsStale(
	hss *HeadersSyncState, prevBlock chainhash.Hash,
) bool {
	var expected chainhash.Hash
	if hss != nil {
		switch hss.Phase() {
		case PhasePresync:
			expected = hss.LastHeaderHash()
		case PhaseRedownload:
			expected = hss.RedownloadTipHash()
		}
	}
	return expected != prevBlock
}

// pushGetHeadersDirect sends a GETHEADERS bypassing the per-peer rate limit.
// Clears pipelinedLocatorHead and refreshes lastGetHeadersTime. Use for
// sends that must go out unconditionally (initial sync, checkpoint-driven,
// inv-driven probe) so the response is not confused with a speculative one.
func (sm *SyncManager) pushGetHeadersDirect(peer *peerpkg.Peer,
	state *peerSyncState, locator blockchain.BlockLocator,
	stopHash *chainhash.Hash, includeCerts bool) error {

	state.pipelinedLocatorHead = nil
	state.lastGetHeadersTime = time.Now()
	return peer.PushGetHeadersMsg(locator, stopHash, includeCerts)
}

// driveRedownloadGetdata drains the REDOWNLOAD Tier-1 FIFO into Tier-2 and
// emits a getdata batch for the newly-popped entries.
func (sm *SyncManager) driveRedownloadGetdata(
	peer *peerpkg.Peer, state *peerSyncState,
	newlyApproved []ApprovedRedownloadEntry,
) {
	hss := state.headersSyncState
	if hss == nil || hss.Phase() != PhaseRedownload {
		return
	}

	free := redownloadPendingCap - len(state.redownloadExpected)
	if free <= 0 {
		if len(newlyApproved) > 0 {
			log.Debugf("REDOWNLOAD getdata back-pressured: peer=%s "+
				"tier2=%d/%d approved_waiting=%d",
				peer.Addr(), len(state.redownloadExpected),
				redownloadPendingCap, hss.RedownloadApprovedLen())
		}
		return
	}

	eligible := hss.EligibleForGetdata()
	popped := hss.PopApprovedRedownloadHashes(min(free, eligible))
	if len(popped) == 0 {
		return
	}

	gdmsg := wire.NewMsgGetDataSizeHint(uint(len(popped)))
	for i := range popped {
		entry := popped[i]
		sm.requestedBlocks[entry.Hash] = struct{}{}
		state.requestedBlocks[entry.Hash] = struct{}{}
		state.redownloadExpected = append(state.redownloadExpected, entry)
		h := entry.Hash
		iv := wire.NewInvVect(wire.InvTypeWitnessBlock, &h)
		_ = gdmsg.AddInvVect(iv)
	}
	if len(gdmsg.InvList) > 0 {
		peer.QueueMessage(gdmsg, nil)
	}
}

// redownloadEntryIndex returns the index of the approved entry matching hash
// inside state.redownloadExpected, or -1 if not found.
func (state *peerSyncState) redownloadEntryIndex(hash chainhash.Hash) int {
	for i := range state.redownloadExpected {
		if state.redownloadExpected[i].Hash == hash {
			return i
		}
	}
	return -1
}

// handleRedownloadBlockArrival handles a BLOCK message approved by an active
// REDOWNLOAD session. Returns handled=true when the block was consumed.
func (sm *SyncManager) handleRedownloadBlockArrival(
	peer *peerpkg.Peer, state *peerSyncState, block *btcutil.Block,
) (handled bool, shouldPunish bool, err error) {

	if state.headersSyncState == nil ||
		state.headersSyncState.Phase() != PhaseRedownload {
		return false, false, nil
	}

	blockHash := block.Hash()
	idx := state.redownloadEntryIndex(*blockHash)
	if idx < 0 {
		return false, false, nil
	}
	entry := state.redownloadExpected[idx]

	delete(state.requestedBlocks, *blockHash)
	delete(sm.requestedBlocks, *blockHash)

	// Cross-check header bytes: the arriving block header must match the
	// header approved during REDOWNLOAD cert-less validation.
	arrivalHeader := block.MsgBlock().BlockHeader()
	if !redownloadHeaderBytesEqual(arrivalHeader, &entry.Header) {
		log.Warnf("REDOWNLOAD peer=%s block %s header bytes "+
			"disagree with approved entry", peer.Addr(), blockHash)
		state.headersSyncState = nil
		state.pipelinedLocatorHead = nil
		sm.resetRedownloadTier2(state)
		return true, true, fmt.Errorf("peer %s sent REDOWNLOAD "+
			"block with mismatched header: %w",
			peer.Addr(), ErrPeerViolation)
	}

	if state.redownloadPendingBlocks == nil {
		state.redownloadPendingBlocks = make(
			map[chainhash.Hash]*btcutil.Block, redownloadPendingCap)
	}
	state.redownloadPendingBlocks[*blockHash] = block

	// Drain in insertion order.
	for len(state.redownloadExpected) > 0 {
		head := state.redownloadExpected[0]
		pending, ok := state.redownloadPendingBlocks[head.Hash]
		if !ok {
			break
		}
		delete(state.redownloadPendingBlocks, head.Hash)
		copy(state.redownloadExpected, state.redownloadExpected[1:])
		state.redownloadExpected[len(state.redownloadExpected)-1] =
			ApprovedRedownloadEntry{}
		state.redownloadExpected = state.redownloadExpected[:len(state.redownloadExpected)-1]

		if err := sm.processRedownloadBlock(peer, state, pending); err != nil {
			log.Warnf("REDOWNLOAD peer=%s rejected block %s: %v",
				peer.Addr(), pending.Hash(), err)
			state.headersSyncState = nil
			state.pipelinedLocatorHead = nil
			sm.resetRedownloadTier2(state)
			return true, true, fmt.Errorf(
				"peer %s REDOWNLOAD block rejected: %w",
				peer.Addr(), err)
		}
	}

	sm.driveRedownloadGetdata(peer, state, nil)
	sm.maybeTriggerRedownloadGetHeaders(peer, state)

	if state.headersSyncState != nil &&
		state.headersSyncState.RedownloadEmissionsComplete() &&
		len(state.redownloadExpected) == 0 {
		log.Infof("REDOWNLOAD peer=%s complete", peer.Addr())
		state.headersSyncState = nil
		state.pipelinedLocatorHead = nil
	}

	return true, false, nil
}

// redownloadHeaderBytesEqual compares two BlockHeaders for byte-equal
// identity via their hashes.
func redownloadHeaderBytesEqual(a, b *wire.BlockHeader) bool {
	return a.BlockHash() == b.BlockHash()
}

// processRedownloadBlock runs chain.ProcessBlock on a REDOWNLOAD block whose
// predecessor has already been accepted.
func (sm *SyncManager) processRedownloadBlock(
	peer *peerpkg.Peer, state *peerSyncState, block *btcutil.Block,
) error {
	_, isOrphan, err := sm.chain.ProcessBlock(block, blockchain.BFNone)
	if err != nil {
		if _, ok := err.(blockchain.RuleError); ok {
			log.Infof("Rejected REDOWNLOAD block %v from %s: %v",
				block.Hash(), peer.Addr(), err)
		} else {
			log.Errorf("Failed to process REDOWNLOAD block %v: %v",
				block.Hash(), err)
		}
		if dbErr, ok := err.(database.Error); ok &&
			dbErr.ErrorCode == database.ErrCorruption {
			panic(dbErr)
		}
		code, reason := mempool.ErrToRejectErr(err)
		peer.PushRejectMsg(wire.CmdBlock, code, reason, block.Hash(), false)
		return err
	}
	if isOrphan {
		log.Warnf("REDOWNLOAD block %v from %s unexpectedly orphan",
			block.Hash(), peer.Addr())
		return nil
	}

	if peer == sm.syncPeer {
		sm.lastProgressTime = time.Now()
	}
	sm.progressLogger.LogBlockHeight(block, sm.chain)
	best := sm.chain.BestSnapshot()
	peer.UpdateLastBlockHeight(best.Height)

	blockHash := block.Hash()
	blockCSI := sm.chain.LookupChainStartInfo(blockHash)
	if blockCSI != nil {
		if sm.shouldDownloadBlocks(blockCSI) {
			if best.Hash == *blockHash {
				state.peerQualityCounter = 0
			}
		} else {
			state.peerQualityCounter++
		}
	}
	return nil
}

// resetRedownloadTier2 discards Tier-2 state after a REDOWNLOAD violation.
func (sm *SyncManager) resetRedownloadTier2(state *peerSyncState) {
	for _, entry := range state.redownloadExpected {
		delete(state.requestedBlocks, entry.Hash)
		delete(sm.requestedBlocks, entry.Hash)
	}
	state.redownloadExpected = nil
	state.redownloadPendingBlocks = nil
}

// maybeTriggerRedownloadGetHeaders issues a follow-up REDOWNLOAD getheaders
// when RequestMore was previously suppressed by Tier-1 saturation and the
// buffer has since drained.
func (sm *SyncManager) maybeTriggerRedownloadGetHeaders(
	peer *peerpkg.Peer, state *peerSyncState,
) {
	hss := state.headersSyncState
	if hss == nil || !hss.ReadyForNextHeaders() {
		return
	}
	if state.pipelinedLocatorHead != nil {
		return
	}
	locator := hss.NextHeadersRequestLocator()
	sm.maybeSendGetHeaders(peer, state, locator, &zeroHash, false)
}

// canDirectFetch returns true if we are close enough to the tip to directly
// fetch blocks from announcements (within antiDoSBufferBlocks of the tip).
func (sm *SyncManager) canDirectFetch() bool {
	best := sm.chain.BestSnapshot()
	targetSpacing := sm.chainParams.TargetTimePerBlock
	maxAge := time.Duration(antiDoSBufferBlocks) * targetSpacing
	return time.Since(best.BlockTime) < maxAge
}

// headerDirectFetchBlocks requests full blocks for headers that have been
// accepted into the block index and are on a chain worth downloading.
func (sm *SyncManager) headerDirectFetchBlocks(
	peer *peerpkg.Peer, state *peerSyncState,
	acceptedHeaders []*blockchain.AcceptedHeader,
) {
	if !sm.canDirectFetch() {
		return
	}
	for _, ah := range acceptedHeaders {
		if _, exists := state.requestedBlocks[ah.Hash]; exists {
			continue
		}
		if _, exists := sm.requestedBlocks[ah.Hash]; exists {
			continue
		}
		iv := wire.NewInvVect(wire.InvTypeWitnessBlock, &ah.Hash)
		limitAdd(state.requestedBlocks, ah.Hash, maxRequestedBlocks)
		limitAdd(sm.requestedBlocks, ah.Hash, maxRequestedBlocks)
		gdmsg := wire.NewMsgGetDataSizeHint(1)
		gdmsg.AddInvVect(iv)
		peer.QueueMessage(gdmsg, nil)
	}
}

// New constructs a new SyncManager. Use Start to begin processing asynchronous
// block, tx, and inv updates.
func New(config *Config) (*SyncManager, error) {
	sm := SyncManager{
		peerNotifier:       config.PeerNotifier,
		chain:              config.Chain,
		txMemPool:          config.TxMemPool,
		chainParams:        config.ChainParams,
		rejectedTxns:       make(map[chainhash.Hash]struct{}),
		requestedTxns:      make(map[chainhash.Hash]struct{}),
		requestedBlocks:    make(map[chainhash.Hash]struct{}),
		peerStates:         make(map[*peerpkg.Peer]*peerSyncState),
		progressLogger:     newBlockProgressLogger("Processed", log),
		msgChan:            make(chan interface{}, config.MaxPeers*3),
		peerVerdicts:       make(chan PeerVerdict, config.MaxPeers),
		headerList:         list.New(),
		quit:               make(chan struct{}),
		feeEstimator:       config.FeeEstimator,
		recentlyFailedSync: make(map[string]time.Time),
	}

	best := sm.chain.BestSnapshot()
	if !config.DisableCheckpoints {
		// Initialize the next checkpoint based on the current height.
		sm.nextCheckpoint = sm.findNextHeaderCheckpoint(best.Height)
		if sm.nextCheckpoint != nil {
			sm.resetHeaderState(&best.Hash, best.Height)
		}
	} else {
		log.Info("Checkpoints are disabled")
	}

	sm.chain.Subscribe(sm.handleBlockchainNotification)

	return &sm, nil
}
