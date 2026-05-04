// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"crypto/rand"
	"math"
	"math/big"
	"time"

	"github.com/aead/siphash"
	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
)

// HeadersSyncPhase represents the current phase of the two-phase presync.
type HeadersSyncPhase int

const (
	PhasePresync HeadersSyncPhase = iota
	PhaseRedownload
	PhaseFinal
)

func (p HeadersSyncPhase) String() string {
	switch p {
	case PhasePresync:
		return "presync"
	case PhaseRedownload:
		return "redownload"
	case PhaseFinal:
		return "final"
	default:
		return "unknown"
	}
}

const (
	// lg2UnauthorisedPresync is the security parameter t: a chain with less
	// than certifiedWorkProportion of its work backed by certificates is
	// rejected with probability >= 1 - 2^{-t}.
	lg2UnauthorisedPresync = 100

	// certifiedWorkProportion is p: the minimum fraction of total work that
	// must be backed by verified certificates for the presync to be secure.
	certifiedWorkProportion = 0.5

	// spotCheckMeanGap is the average gap between mandatory periodic spot
	// checks. Mandatory checks are drawn from U[1, 2*spotCheckMeanGap].
	spotCheckMeanGap = 1000

	// redownloadApprovedCap is the Tier-1 REDOWNLOAD approved-headers FIFO cap.
	redownloadApprovedCap = 500

	// redownloadApprovedHeadroom is the minimum free capacity required in
	// Tier-1 before ProcessNextHeaders sets RequestMore=true, ensuring there is
	// always room for at least two full batches of incoming REDOWNLOAD headers.
	redownloadApprovedHeadroom = 2 * wire.MaxBlockHeadersPerMsg

	// redownloadGetdataDepth is the minimum number of commitment-checked
	// REDOWNLOAD headers that must exist in Tier-1 on top of an entry before
	// it is eligible for getdata.
	redownloadGetdataDepth = lg2UnauthorisedPresync
)

// chainStartInfo captures the immutable state about the fork point for the
// presync session.
type chainStartInfo struct {
	blockchain.ChainStartInfo
	// locator holds exponentially-spaced ancestor hashes from chain_start
	// back towards genesis, used as fallback entries in getheaders locators.
	locator []*chainhash.Hash
}

// ApprovedRedownloadEntry holds an approved cert-less header from the
// REDOWNLOAD phase. Tier-1 of the two-tier REDOWNLOAD buffer stores a FIFO of
// these entries; Tier-2 moves them into getdata-in-flight or pending-block
// status.
type ApprovedRedownloadEntry struct {
	Hash   chainhash.Hash
	Header wire.BlockHeader
}

// pendingSpotCheck represents a spot-check target that has been issued to the
// peer but whose cert-bearing response has not yet arrived.
type pendingSpotCheck struct {
	height    int32
	hash      chainhash.Hash
	prevBlock chainhash.Hash
}

// SpotCheckRequest is a getheaders request to be sent for a spot-check.
// The locator's first entry is the target's parent hash; StopHash is the
// target's own hash, ensuring the response contains exactly that one header.
type SpotCheckRequest struct {
	Locator  []*chainhash.Hash
	StopHash chainhash.Hash
}

// redownloadCursor tracks the REDOWNLOAD phase tip: the hash, timestamp, and
// height of the last header successfully appended to Tier-1. It is initialized
// from chainStart at the PRESYNC→REDOWNLOAD transition and advanced
// exclusively in validateAndStoreRedownloadedHeader. It is never reset by
// PopApprovedRedownloadHashes, which means it stays valid even while Tier-1 is
// being drained into Tier-2.
type redownloadCursor struct {
	hash      chainhash.Hash
	timestamp int64
	height    int32
}

// HeadersSyncResult is the result returned by ProcessNextHeaders.
type HeadersSyncResult struct {
	// NewlyApproved lists the cert-less REDOWNLOAD entries just appended to
	// Tier-1 by this call. Empty in PRESYNC phases.
	NewlyApproved []ApprovedRedownloadEntry

	// SpotCheckRequests lists getheaders requests to send for newly queued
	// spot checks. Each request uses IncludeCertificates=true and a stop
	// hash targeting exactly one header.
	SpotCheckRequests []SpotCheckRequest

	Success      bool
	RequestMore  bool
	ShouldPunish bool
}

// HeadersSyncState implements the two-phase headers presync state machine.
// See doc/pearl-headers-block-sync.md for the full specification.
type HeadersSyncState struct {
	peerID int32

	chainParams *chaincfg.Params

	chainStart          chainStartInfo
	minimumRequiredWork *big.Int
	chainStartNextNBits uint32

	// workNormalization is C = t * ln(2) / ((1-p) * remainingWork), where
	// remainingWork = minimumRequiredWork - chainStart.WorkSum. Each header
	// is spot-checked with probability min(C * header_work, 1).
	workNormalization float64

	// hashSalt is the 128-bit session key for commitment bit computation
	// (SipHash-2-4). Drawn from crypto/rand once per session.
	hashSalt [16]byte

	phase HeadersSyncPhase

	lastHeaderReceived wire.BlockHeader
	lastHeaderHash     chainhash.Hash
	currentHeight      int32
	currentChainWork   *big.Int
	nextExpectedNBits  uint32

	headerCommitments *BitDeque

	// nextSpotCheckHeight is the absolute height of the next mandatory
	// periodic spot-check during PRESYNC. Drawn uniformly such that the
	// gap since the previous anchor is in [1, 2*spotCheckMeanGap].
	nextSpotCheckHeight int32

	// pendingSpotChecks is the queue of spot-check targets for which a
	// getheaders (with IncludeCertificates=true and stop hash) has been
	// issued but no response has been received yet. Multiple spot checks
	// can be in-flight simultaneously. Backpressure prevents headers from
	// advancing more than spotCheckMeanGap heights ahead of the oldest
	// pending entry.
	pendingSpotChecks []pendingSpotCheck

	// redownloadApproved is the Tier-1 FIFO of cert-less headers approved
	// during the REDOWNLOAD phase, bounded by redownloadApprovedCap.
	// SyncManager pops entries via PopApprovedRedownloadHashes and issues
	// getdata for them (Tier-2).
	redownloadApproved []ApprovedRedownloadEntry

	// redownloadCursor is the REDOWNLOAD tip cursor. Initialized from
	// chainStart at PRESYNC→REDOWNLOAD, advanced per-header in
	// validateAndStoreRedownloadedHeader, never modified by Tier-1 drain.
	redownloadCursor redownloadCursor

	redownloadChainWork        *big.Int
	processAllRemainingHeaders bool
	redownloadShortBatchSeen   bool

	shouldPunish bool
}

// NewHeadersSyncState creates a new presync state machine for a peer.
func NewHeadersSyncState(
	peerID int32,
	params *chaincfg.Params,
	start chainStartInfo,
	minimumRequiredWork *big.Int,
) *HeadersSyncState {

	start.WorkSum = new(big.Int).Set(start.WorkSum)

	s := &HeadersSyncState{
		peerID:              peerID,
		chainParams:         params,
		chainStart:          start,
		minimumRequiredWork: new(big.Int).Set(minimumRequiredWork),
		phase:               PhasePresync,
		currentHeight:       start.Height,
		currentChainWork:    new(big.Int).Set(start.WorkSum),
		headerCommitments:   NewBitDeque(1024),
		redownloadChainWork: new(big.Int),
	}

	// Random salt for commitment hashing (SipHash-2-4 key).
	rand.Read(s.hashSalt[:])

	// First spot check lies in [start.Height+1, start.Height+2*spotCheckMeanGap].
	s.scheduleNextSpotCheck(start.Height)

	// Initialize lastHeaderReceived with chain_start fields for timestamp/bits
	// lookups, and lastHeaderHash with the actual chain_start hash for
	// continuity checks and locator construction.
	s.lastHeaderReceived = wire.BlockHeader{
		Bits:      start.Bits,
		Timestamp: time.Unix(start.Timestamp, 0),
	}
	s.lastHeaderHash = start.Hash

	s.chainStartNextNBits = s.computeNextNBits(
		start.Height, start.Bits,
		start.Timestamp, start.PrevTimestamp,
	)
	s.nextExpectedNBits = s.chainStartNextNBits

	remainingWork := new(big.Int).Sub(s.minimumRequiredWork, s.chainStart.WorkSum)
	if remainingWork.Sign() > 0 {
		rw, _ := new(big.Float).SetInt(remainingWork).Float64()
		s.workNormalization = float64(lg2UnauthorisedPresync) * math.Ln2 /
			((1.0 - certifiedWorkProportion) * rw)
	}

	log.Infof("Headers presync started with peer=%d: height=%d, "+
		"min_work=%s, work_norm=%e",
		peerID, s.currentHeight,
		s.minimumRequiredWork, s.workNormalization)

	return s
}

// Phase returns the current phase.
func (s *HeadersSyncState) Phase() HeadersSyncPhase { return s.phase }

// LastHeaderHash returns the hash of the last header processed in PRESYNC.
func (s *HeadersSyncState) LastHeaderHash() chainhash.Hash {
	return s.lastHeaderHash
}

// RedownloadTipHash returns the hash stored in the REDOWNLOAD cursor.
func (s *HeadersSyncState) RedownloadTipHash() chainhash.Hash {
	return s.redownloadCursor.hash
}

// SpotCheckBackpressured returns true when the PRESYNC header pipeline should
// pause because the current height is more than spotCheckMeanGap ahead of the
// oldest unresolved spot check.
func (s *HeadersSyncState) SpotCheckBackpressured() bool {
	return len(s.pendingSpotChecks) > 0 &&
		s.currentHeight-s.pendingSpotChecks[0].height >= spotCheckMeanGap
}

// SpeculativeLocator builds a getheaders locator with tip as the first entry
// and the chainStart exponential ancestor list appended.
func (s *HeadersSyncState) SpeculativeLocator(tip chainhash.Hash) []*chainhash.Hash {
	locator := make([]*chainhash.Hash, 0, 1+len(s.chainStart.locator))
	tipCopy := tip
	locator = append(locator, &tipCopy)
	locator = append(locator, s.chainStart.locator...)
	return locator
}

// PopApprovedRedownloadHashes removes up to n entries from the front of the
// Tier-1 FIFO and returns them. Entries remain in redownloadApproved until
// this call; the cursor is not affected.
func (s *HeadersSyncState) PopApprovedRedownloadHashes(n int) []ApprovedRedownloadEntry {
	if n <= 0 || len(s.redownloadApproved) == 0 {
		return nil
	}
	n = min(n, len(s.redownloadApproved))
	out := s.redownloadApproved[:n:n]
	s.redownloadApproved = s.redownloadApproved[n:]
	return out
}

// RedownloadApprovedLen returns the number of entries currently in Tier-1.
func (s *HeadersSyncState) RedownloadApprovedLen() int {
	return len(s.redownloadApproved)
}

// EligibleForGetdata returns the number of Tier-1 entries that may be
// promoted to Tier-2 right now. Entries are held back until at least
// redownloadGetdataDepth more commitment-checked headers sit on top of them.
// When the headers phase is done (short batch seen), all remaining entries
// are immediately eligible.
func (s *HeadersSyncState) EligibleForGetdata() int {
	if s.redownloadShortBatchSeen {
		return len(s.redownloadApproved)
	}
	n := len(s.redownloadApproved) - redownloadGetdataDepth
	if n < 0 {
		return 0
	}
	return n
}

// hasRedownloadFifoCapacity reports whether Tier-1 has room for at least
// redownloadApprovedHeadroom more entries.
func (s *HeadersSyncState) hasRedownloadFifoCapacity() bool {
	return len(s.redownloadApproved)+redownloadApprovedHeadroom <= redownloadApprovedCap
}

// ReadyForNextHeaders returns true when the REDOWNLOAD phase has capacity for
// more headers (Tier-1 is not saturated) and the session is not yet complete.
// Used by maybeTriggerRedownloadGetHeaders to re-arm the getheaders pipeline
// after Tier-1 drains.
func (s *HeadersSyncState) ReadyForNextHeaders() bool {
	if s.phase != PhaseRedownload {
		return false
	}
	if s.redownloadShortBatchSeen {
		return false
	}
	return s.hasRedownloadFifoCapacity()
}

// RedownloadEmissionsComplete returns true when the REDOWNLOAD headers phase
// is done (a short batch was seen after processAllRemainingHeaders) and Tier-1
// is empty. SyncManager uses this together with Tier-2 emptiness to decide
// when to tear down the session.
func (s *HeadersSyncState) RedownloadEmissionsComplete() bool {
	return s.redownloadShortBatchSeen && len(s.redownloadApproved) == 0
}

// ProcessNextHeaders processes a batch of headers from the peer.
// fullMessage is true when the peer filled the batch to wire.MaxBlockHeadersPerMsg,
// indicating more headers are available; false signals a partial (final) batch.
func (s *HeadersSyncState) ProcessNextHeaders(
	headers []wire.MsgHeader, fullMessage bool,
) HeadersSyncResult {

	var result HeadersSyncResult
	s.shouldPunish = false

	if len(headers) == 0 || s.phase == PhaseFinal {
		return result
	}

	if s.phase == PhasePresync {
		if scIdx := s.findPendingSpotCheck(headers[0].BlockHeader.BlockHash()); scIdx >= 0 {
			hwc := headers[0]
			scHeight := s.pendingSpotChecks[scIdx].height
			if hwc.BlockCertificate() == nil {
				log.Infof("Headers presync aborted with peer=%d: "+
					"spot-check cert missing (presync)", s.peerID)
			} else if err := zkpow.VerifyCertificate(&hwc.BlockHeader, hwc.BlockCertificate()); err != nil {
				log.Warnf("Headers presync aborted with peer=%d: "+
					"spot-check cert invalid: %v (presync)", s.peerID, err)
				s.shouldPunish = true
			} else {
				if scIdx == 0 {
					s.pendingSpotChecks = s.pendingSpotChecks[1:]
				} else {
					s.pendingSpotChecks = append(s.pendingSpotChecks[:scIdx], s.pendingSpotChecks[scIdx+1:]...)
				}
				result.Success = true

				log.Infof("Headers presync spot-check passed with peer=%d: "+
					"height=%d, pending=%d (presync)",
					s.peerID, scHeight, len(s.pendingSpotChecks))

				if s.currentChainWork.Cmp(s.minimumRequiredWork) >= 0 &&
					len(s.pendingSpotChecks) == 0 {
					s.transitionToRedownload()
					result.RequestMore = true
					log.Infof("Headers presync transition with peer=%d: "+
						"sufficient work at height=%d, redownloading from height=%d",
						s.peerID, s.currentHeight, s.redownloadCursor.height)
				} else if !s.SpotCheckBackpressured() && !s.PresyncWorkSufficient() {
					result.RequestMore = true
				}
			}
			result.ShouldPunish = s.shouldPunish
			if !result.Success {
				s.finalize()
			}
			return result
		}
	}

	switch s.phase {
	case PhasePresync:
		prevPending := len(s.pendingSpotChecks)
		result.Success = s.validateAndStoreCommitments(headers)
		if result.Success {
			for _, sc := range s.pendingSpotChecks[prevPending:] {
				result.SpotCheckRequests = append(result.SpotCheckRequests, SpotCheckRequest{
					Locator:  s.SpeculativeLocator(sc.prevBlock),
					StopHash: sc.hash,
				})
			}
			switch {
			case s.phase == PhaseRedownload:
				result.RequestMore = true
			case fullMessage:
				if !s.SpotCheckBackpressured() && !s.PresyncWorkSufficient() {
					result.RequestMore = true
				}
				log.Infof("Headers presync with peer=%d: "+
					"height=%d, commitments=%d, pending_checks=%d (presync)",
					s.peerID, s.currentHeight,
					s.headerCommitments.Len(),
					len(s.pendingSpotChecks))
			default:
				log.Infof("Headers presync aborted with peer=%d: "+
					"incomplete message at height=%d (presync)", s.peerID, s.currentHeight)
			}
		}

	case PhaseRedownload:
		before := len(s.redownloadApproved)
		result.Success = true
		for i := range headers {
			if !s.validateAndStoreRedownloadedHeader(&headers[i]) {
				result.Success = false
				break
			}
		}

		if result.Success {
			if added := len(s.redownloadApproved) - before; added > 0 {
				result.NewlyApproved = append(
					make([]ApprovedRedownloadEntry, 0, added),
					s.redownloadApproved[before:]...,
				)
			}

			if !fullMessage {
				if s.processAllRemainingHeaders {
					s.redownloadShortBatchSeen = true
					log.Infof("Headers presync complete with peer=%d: "+
						"short batch at height=%d (redownload)",
						s.peerID, s.redownloadCursor.height)
				} else {
					log.Infof("Headers presync aborted with peer=%d: "+
						"incomplete message at height=%d (redownload)",
						s.peerID, s.redownloadCursor.height)
					result.Success = false
				}
			} else if s.hasRedownloadFifoCapacity() {
				result.RequestMore = true
				log.Infof("Headers presync redownload progress with peer=%d: "+
					"height=%d, approved=%d (redownload)",
					s.peerID, s.redownloadCursor.height, len(s.redownloadApproved))
			}
		}
	}

	result.ShouldPunish = s.shouldPunish

	// Finalize for non-REDOWNLOAD failure paths and all PRESYNC failures.
	// REDOWNLOAD stays alive (phase remains PhaseRedownload) until the
	// SyncManager explicitly tears it down after both Tier-1 and Tier-2 drain.
	if !result.Success && s.phase != PhaseRedownload {
		s.finalize()
	} else if result.Success && s.phase != PhaseRedownload && !result.RequestMore &&
		!s.PresyncWorkSufficient() {
		s.finalize()
	}
	return result
}

// NextHeadersRequestLocator builds the block locator for the next getheaders.
// The locator starts with the state-specific tip hash, followed by the
// exponentially-spaced ancestors of chain_start (matching Bitcoin Core's
// LocatorEntries).
func (s *HeadersSyncState) NextHeadersRequestLocator() []*chainhash.Hash {
	if s.phase == PhaseFinal {
		return nil
	}
	var tipHash chainhash.Hash
	switch s.phase {
	case PhasePresync:
		tipHash = s.lastHeaderHash
	case PhaseRedownload:
		tipHash = s.redownloadCursor.hash
	}
	return s.SpeculativeLocator(tipHash)
}

// findPendingSpotCheck returns the index of the pending spot check whose hash
// matches h, or -1 if not found. Checks the front first since responses
// typically arrive in order.
func (s *HeadersSyncState) findPendingSpotCheck(h chainhash.Hash) int {
	if len(s.pendingSpotChecks) > 0 && s.pendingSpotChecks[0].hash == h {
		return 0
	}
	for i := 1; i < len(s.pendingSpotChecks); i++ {
		if s.pendingSpotChecks[i].hash == h {
			return i
		}
	}
	return -1
}

func (s *HeadersSyncState) transitionToRedownload() {
	s.redownloadApproved = s.redownloadApproved[:0]
	s.redownloadCursor = redownloadCursor{
		hash:      s.chainStart.Hash,
		timestamp: s.chainStart.Timestamp,
		height:    s.chainStart.Height,
	}
	s.redownloadChainWork = new(big.Int).Set(s.chainStart.WorkSum)
	s.nextExpectedNBits = s.chainStartNextNBits
	s.phase = PhaseRedownload
}

func (s *HeadersSyncState) finalize() {
	s.headerCommitments.Clear()
	s.redownloadApproved = nil
	s.processAllRemainingHeaders = false
	s.redownloadShortBatchSeen = false
	s.pendingSpotChecks = nil
	s.currentHeight = 0
	s.phase = PhaseFinal
}

// shouldSpotCheckByWork returns true with probability min(C * headerWork, 1),
// implementing the work-proportional spot-check trigger.
func (s *HeadersSyncState) shouldSpotCheckByWork(headerWork *big.Int) bool {
	if s.workNormalization == 0 {
		return false
	}
	wf, _ := new(big.Float).SetInt(headerWork).Float64()
	prob := s.workNormalization * wf
	if prob >= 1.0 {
		return true
	}
	var buf [2]byte
	rand.Read(buf[:])
	r := uint16(buf[0]) | uint16(buf[1])<<8
	return r < uint16(prob*65536)
}

// PresyncWorkSufficient returns true when cumulative work has reached the
// minimum required threshold but the REDOWNLOAD transition has not yet
// occurred (pending spot checks may be blocking it).
func (s *HeadersSyncState) PresyncWorkSufficient() bool {
	return s.phase == PhasePresync &&
		s.currentChainWork.Cmp(s.minimumRequiredWork) >= 0
}

// scheduleNextSpotCheck sets nextSpotCheckHeight to baseHeight + U[1, 2*spotCheckMeanGap],
// bounding the worst-case distance between consecutive mandatory spot-checks to
// 2*spotCheckMeanGap headers while keeping the mean gap at spotCheckMeanGap.
func (s *HeadersSyncState) scheduleNextSpotCheck(baseHeight int32) {
	n, err := rand.Int(rand.Reader, big.NewInt(2*spotCheckMeanGap))
	if err != nil {
		panic("headers presync: crypto/rand failed: " + err.Error())
	}
	s.nextSpotCheckHeight = baseHeight + 1 + int32(n.Int64())
}

func (s *HeadersSyncState) validateAndStoreCommitments(headers []wire.MsgHeader) bool {
	if s.phase != PhasePresync {
		return false
	}

	if headers[0].BlockHeader.PrevBlock != s.lastHeaderHash {
		log.Infof("Headers presync aborted with peer=%d: "+
			"non-continuous at height=%d (presync)", s.peerID, s.currentHeight)
		return false
	}

	for i := range headers {
		if s.currentChainWork.Cmp(s.minimumRequiredWork) >= 0 {
			break
		}

		if !s.validateAndProcessSingleHeader(&headers[i]) {
			return false
		}

		spotCheck := s.currentHeight == s.nextSpotCheckHeight ||
			s.shouldSpotCheckByWork(blockchain.CalcWork(headers[i].BlockHeader.Bits))

		if spotCheck {
			hwc := &headers[i]
			s.pendingSpotChecks = append(s.pendingSpotChecks, pendingSpotCheck{
				height:    s.currentHeight,
				hash:      s.lastHeaderHash,
				prevBlock: hwc.BlockHeader.PrevBlock,
			})
			s.scheduleNextSpotCheck(s.currentHeight)
		}
	}

	if s.currentChainWork.Cmp(s.minimumRequiredWork) >= 0 {
		if len(s.pendingSpotChecks) > 0 {
			return true
		}
		s.transitionToRedownload()
		log.Infof("Headers presync transition with peer=%d: "+
			"sufficient work at height=%d, redownloading from height=%d",
			s.peerID, s.currentHeight, s.redownloadCursor.height)
	}
	return true
}

func (s *HeadersSyncState) validateAndProcessSingleHeader(hwc *wire.MsgHeader) bool {
	if s.phase != PhasePresync {
		return false
	}
	header := &hwc.BlockHeader
	nextHeight := s.currentHeight + 1

	if !s.checkHeaderTransition(header, s.lastHeaderReceived.Timestamp.Unix(), nextHeight, nil) {
		return false
	}

	var zeroHash chainhash.Hash
	if header.ProofCommitment == zeroHash {
		log.Warnf("Headers presync aborted with peer=%d: "+
			"zero ProofCommitment at height=%d (presync)", s.peerID, nextHeight)
		s.shouldPunish = true
		return false
	}

	headerHash := header.BlockHash()

	bit := s.commitBit(headerHash)
	s.headerCommitments.PushBack(bit)

	work := blockchain.CalcWork(header.Bits)
	s.currentChainWork = new(big.Int).Add(s.currentChainWork, work)

	s.nextExpectedNBits = s.computeNextNBits(
		nextHeight, header.Bits,
		header.Timestamp.Unix(), s.lastHeaderReceived.Timestamp.Unix(),
	)
	s.lastHeaderReceived = *header
	s.lastHeaderHash = headerHash
	s.currentHeight = nextHeight
	return true
}

func (s *HeadersSyncState) checkHeaderTransition(
	header *wire.BlockHeader, lastTime int64, nextHeight int32,
	prevHash *chainhash.Hash,
) bool {
	phaseName := s.phase.String()

	if prevHash != nil && header.PrevBlock != *prevHash {
		log.Infof("Headers presync aborted with peer=%d: "+
			"non-continuous at height=%d (%s)", s.peerID, nextHeight, phaseName)
		return false
	}

	if header.Timestamp.Unix() < lastTime+int64(blockchain.MinTimestampDeltaSeconds) {
		log.Warnf("Headers presync aborted with peer=%d: "+
			"timestamp not increasing at height=%d (%s)", s.peerID, nextHeight, phaseName)
		s.shouldPunish = true
		return false
	}

	if header.Bits != s.nextExpectedNBits {
		if s.chainParams.ReduceMinDifficulty &&
			header.Timestamp.Unix() > lastTime+int64(s.chainParams.MinDiffReductionTime/time.Second) &&
			header.Bits == s.chainParams.PowLimitBits {
			// Testnet min-difficulty escape hatch.
		} else {
			log.Warnf("Headers presync aborted with peer=%d: "+
				"invalid difficulty at height=%d (%s)", s.peerID, nextHeight, phaseName)
			s.shouldPunish = true
			return false
		}
	}

	return true
}

func (s *HeadersSyncState) validateAndStoreRedownloadedHeader(hwc *wire.MsgHeader) bool {
	if s.phase != PhaseRedownload {
		return false
	}
	header := &hwc.BlockHeader
	nextHeight := s.redownloadCursor.height + 1

	prevHash := s.redownloadCursor.hash
	lastTime := s.redownloadCursor.timestamp
	if !s.checkHeaderTransition(header, lastTime, nextHeight, &prevHash) {
		return false
	}

	var zeroHash chainhash.Hash
	if header.ProofCommitment == zeroHash {
		log.Warnf("Headers presync aborted with peer=%d: "+
			"zero ProofCommitment at height=%d (redownload)", s.peerID, nextHeight)
		s.shouldPunish = true
		return false
	}

	if !s.processAllRemainingHeaders {
		if s.headerCommitments.Empty() {
			log.Infof("Headers presync aborted with peer=%d: "+
				"commitment overrun at height=%d (redownload)", s.peerID, nextHeight)
			return false
		}
		headerHash := header.BlockHash()
		bit := s.commitBit(headerHash)
		expected := s.headerCommitments.PopFront()
		if bit != expected {
			log.Infof("Headers presync aborted with peer=%d: "+
				"commitment mismatch at height=%d (redownload)", s.peerID, nextHeight)
			return false
		}
	}

	work := blockchain.CalcWork(header.Bits)
	s.redownloadChainWork = new(big.Int).Add(s.redownloadChainWork, work)
	if s.redownloadChainWork.Cmp(s.minimumRequiredWork) >= 0 {
		s.processAllRemainingHeaders = true
	}

	headerHash := header.BlockHash()
	s.redownloadApproved = append(s.redownloadApproved, ApprovedRedownloadEntry{
		Hash:   headerHash,
		Header: *header,
	})

	s.redownloadCursor = redownloadCursor{
		hash:      headerHash,
		timestamp: header.Timestamp.Unix(),
		height:    nextHeight,
	}

	s.nextExpectedNBits = s.computeNextNBits(
		nextHeight, header.Bits,
		header.Timestamp.Unix(), lastTime,
	)
	return true
}

// commitBit computes the 1-bit commitment for a header hash using
// SipHash-2-4 keyed by the per-session hashSalt. The PRF property ensures
// an attacker without the salt cannot predict or bias bits for arbitrary
// headers.
func (s *HeadersSyncState) commitBit(hash chainhash.Hash) bool {
	return siphash.Sum64(hash[:], &s.hashSalt)&1 != 0
}

func (s *HeadersSyncState) computeNextNBits(height int32, bits uint32, ts, prevTs int64) uint32 {
	result, err := blockchain.CalcNextRequiredDifficultyFromValues(
		s.chainParams, height, bits, ts, prevTs,
	)
	if err != nil {
		return s.chainParams.PowLimitBits
	}
	return result
}
