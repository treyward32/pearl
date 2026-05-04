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
	PhasePresyncSpotCheck
	PhaseRedownload
	PhaseFinal
)

func (p HeadersSyncPhase) String() string {
	switch p {
	case PhasePresync:
		return "presync"
	case PhasePresyncSpotCheck:
		return "presync spot-check"
	case PhaseRedownload:
		return "redownload"
	case PhaseFinal:
		return "final"
	default:
		return "unknown"
	}
}

const (
	spotCheckInvProb = 1000

	// redownloadApprovedCap is the Tier-1 REDOWNLOAD approved-headers FIFO cap.
	redownloadApprovedCap = 500

	// redownloadApprovedHeadroom is the minimum free capacity required in
	// Tier-1 before ProcessNextHeaders sets RequestMore=true, ensuring there is
	// always room for at least two full batches of incoming REDOWNLOAD headers.
	redownloadApprovedHeadroom = 2 * wire.MaxBlockHeadersPerMsg

	// redownloadGetdataDepth is the minimum number of commitment-checked
	// REDOWNLOAD headers that must exist in Tier-1 on top of an entry before
	// it is eligible for getdata. This ensures no block is requested until at
	// least one full batch of commitment-verified headers follows it.
	redownloadGetdataDepth = wire.MaxBlockHeadersPerMsg
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
	// Tier-1 by this call. Empty in PRESYNC and PRESYNC_SPOT_CHECK phases.
	NewlyApproved []ApprovedRedownloadEntry

	Success      bool
	RequestMore  bool
	ShouldPunish bool
}

// HeadersSyncState implements the two-phase headers presync state machine.
// See /home/ohadk/workspace/pearl/doc/pearl-headers-block-sync.md for the full specification.
type HeadersSyncState struct {
	peerID int32

	chainParams *chaincfg.Params

	chainStart          chainStartInfo
	minimumRequiredWork *big.Int
	certThresholdNBits  uint32
	chainStartNextNBits uint32

	maxCommitments uint64

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

	spotCheckHeader *wire.BlockHeader

	// nextSpotCheckHeight is the absolute height of the next header to
	// spot-check during cert-less PRESYNC. Drawn uniformly such that the
	// gap since the previous anchor is in [0, 2*spotCheckInvProb), which
	// bounds the worst-case distance between consecutive spot-checks.
	nextSpotCheckHeight int32

	// --- REDOWNLOAD Tier-1 state ---

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
//
// peerID: peer identifier for logging
// params: chain configuration
// start: info about the fork point (the last known block before the peer's chain)
// minimumRequiredWork: the anti-DoS work threshold
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

	// First spot check lies in [start.Height+1, start.Height+2*spotCheckInvProb].
	s.armNextSpotCheck(start.Height)

	// Initialize lastHeaderReceived with chain_start fields for timestamp/bits
	// lookups, and lastHeaderHash with the actual chain_start hash for
	// continuity checks and locator construction.
	s.lastHeaderReceived = wire.BlockHeader{
		Bits:      start.Bits,
		Timestamp: time.Unix(start.Timestamp, 0),
	}
	s.lastHeaderHash = start.Hash

	// Compute initial expected nBits.
	s.chainStartNextNBits = s.computeNextNBits(
		start.Height, start.Bits,
		start.Timestamp, start.PrevTimestamp,
	)
	s.nextExpectedNBits = s.chainStartNextNBits

	s.maxCommitments = uint64(s.maxBlocksSinceStart())
	s.certThresholdNBits = s.computeCertThreshold()

	log.Infof("Headers presync started with peer=%d: height=%d, "+
		"max_commitments=%d, min_work=%s, cert_threshold=0x%08x",
		peerID, s.currentHeight, s.maxCommitments,
		s.minimumRequiredWork, s.certThresholdNBits)

	return s
}

// Phase returns the current phase.
func (s *HeadersSyncState) Phase() HeadersSyncPhase { return s.phase }

// PresyncHeight returns the height reached during PRESYNC.
func (s *HeadersSyncState) PresyncHeight() int32 { return s.currentHeight }

// PresyncWork returns the cumulative claimed work so far.
func (s *HeadersSyncState) PresyncWork() *big.Int { return s.currentChainWork }

// CurrentHeight returns the phase-aware current height:
// - In PRESYNC / PRESYNC_SPOT_CHECK / FINAL: the PRESYNC endpoint height.
// - In REDOWNLOAD: the REDOWNLOAD cursor height (last header appended to Tier-1).
func (s *HeadersSyncState) CurrentHeight() int32 {
	if s.phase == PhaseRedownload {
		return s.redownloadCursor.height
	}
	return s.currentHeight
}

// RedownloadHeight returns the height of the REDOWNLOAD cursor.
func (s *HeadersSyncState) RedownloadHeight() int32 {
	return s.redownloadCursor.height
}

// LastHeaderHash returns the hash of the last header processed in PRESYNC.
func (s *HeadersSyncState) LastHeaderHash() chainhash.Hash {
	return s.lastHeaderHash
}

// RedownloadTipHash returns the hash stored in the REDOWNLOAD cursor.
func (s *HeadersSyncState) RedownloadTipHash() chainhash.Hash {
	return s.redownloadCursor.hash
}

// ShouldIncludeCertificates returns whether the next getheaders request
// should ask for certificates.
func (s *HeadersSyncState) ShouldIncludeCertificates() bool {
	switch s.phase {
	case PhasePresync:
		return isAtLeastAsHard(s.lastHeaderReceived.Bits, s.certThresholdNBits)
	case PhaseRedownload, PhaseFinal:
		return false
	default:
		return true
	}
}

// ShouldIncludeCertificatesAfterBits predicts the cert flag for the speculative
// next GETHEADERS, using bits as a proxy for the next batch's difficulty
// without waiting for ProcessNextHeaders to complete. This mirrors the logic
// in ShouldIncludeCertificates for PhasePresync; REDOWNLOAD always returns
// false regardless of bits.
func (s *HeadersSyncState) ShouldIncludeCertificatesAfterBits(bits uint32) bool {
	if s.phase != PhasePresync {
		return false
	}
	return isAtLeastAsHard(bits, s.certThresholdNBits)
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
	if n > len(s.redownloadApproved) {
		n = len(s.redownloadApproved)
	}
	out := make([]ApprovedRedownloadEntry, n)
	copy(out, s.redownloadApproved[:n])
	// Compact in-place.
	remaining := len(s.redownloadApproved) - n
	copy(s.redownloadApproved, s.redownloadApproved[n:])
	for i := remaining; i < len(s.redownloadApproved); i++ {
		s.redownloadApproved[i] = ApprovedRedownloadEntry{}
	}
	s.redownloadApproved = s.redownloadApproved[:remaining]
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
func (s *HeadersSyncState) ProcessNextHeaders(
	headers []wire.MsgHeader, fullMessage bool,
) HeadersSyncResult {

	var result HeadersSyncResult
	s.shouldPunish = false

	if len(headers) == 0 || s.phase == PhaseFinal {
		return result
	}

	switch s.phase {
	case PhasePresync:
		result.Success = s.validateAndStoreCommitments(headers)
		if result.Success {
			if fullMessage || s.phase == PhaseRedownload || s.phase == PhasePresyncSpotCheck {
				result.RequestMore = true
				if s.phase == PhasePresync {
					log.Infof("Headers presync with peer=%d: "+
						"height=%d, commitments=%d/%d, certs_required=%t (presync)",
						s.peerID, s.currentHeight,
						s.headerCommitments.Len(), s.maxCommitments,
						s.ShouldIncludeCertificates())
				}
			} else {
				log.Infof("Headers presync aborted with peer=%d: "+
					"incomplete message at height=%d (presync)", s.peerID, s.currentHeight)
			}
		}

	case PhasePresyncSpotCheck:
		hwc := headers[0]
		if hwc.BlockHeader.BlockHash() != s.spotCheckHeader.BlockHash() {
			log.Infof("Headers presync aborted with peer=%d: "+
				"spot-check hash mismatch (presync)", s.peerID)
		} else if hwc.BlockCertificate() == nil {
			log.Infof("Headers presync aborted with peer=%d: "+
				"spot-check cert missing (presync)", s.peerID)
		} else if err := zkpow.VerifyCertificate(&hwc.BlockHeader, hwc.BlockCertificate()); err != nil {
			log.Warnf("Headers presync aborted with peer=%d: "+
				"spot-check cert invalid: %v (presync)", s.peerID, err)
			s.shouldPunish = true
		} else {
			s.spotCheckHeader = nil
			s.phase = PhasePresync
			s.armNextSpotCheck(s.currentHeight)
			result.Success = true
			result.RequestMore = true
			log.Infof("Headers presync spot-check passed with peer=%d: "+
				"resuming at height=%d (presync)", s.peerID, s.currentHeight)
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
	} else if result.Success && s.phase != PhaseRedownload && !result.RequestMore {
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
	case PhasePresyncSpotCheck:
		tipHash = s.spotCheckHeader.PrevBlock
	case PhaseRedownload:
		tipHash = s.redownloadCursor.hash
	}
	return s.SpeculativeLocator(tipHash)
}

// --- internal methods ---

func (s *HeadersSyncState) finalize() {
	s.headerCommitments.Clear()
	s.redownloadApproved = nil
	s.processAllRemainingHeaders = false
	s.redownloadShortBatchSeen = false
	s.spotCheckHeader = nil
	s.currentHeight = 0
	s.phase = PhaseFinal
}

// armNextSpotCheck schedules the next cert-less PRESYNC spot-check by drawing
// a uniform gap in [1, 2*spotCheckInvProb+1] and anchoring it at baseHeight.
// This keeps the mean gap at spotCheckInvProb while bounding the worst-case
// distance between consecutive spot-checks to 2*spotCheckInvProb - 1 headers.
func (s *HeadersSyncState) armNextSpotCheck(baseHeight int32) {
	n, err := rand.Int(rand.Reader, big.NewInt(2*spotCheckInvProb))
	if err != nil {
		// crypto/rand.Reader is documented never to return an error on
		// supported platforms; panic is the only sane response.
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

	includeCerts := s.ShouldIncludeCertificates()

	// Determine whether this batch covers the pre-scheduled spot-check
	// target height. Heights covered by this batch are
	// [s.currentHeight+1, s.currentHeight+len(headers)] at entry.
	targetIdx := -1
	if d := int64(s.nextSpotCheckHeight) - int64(s.currentHeight); d >= 1 && d <= int64(len(headers)) {
		targetIdx = int(d - 1)
	}

	for i := range headers {
		if !s.validateAndProcessSingleHeader(&headers[i], includeCerts) {
			return false
		}
	}

	if s.currentChainWork.Cmp(s.minimumRequiredWork) >= 0 {
		// Transition to REDOWNLOAD: initialize the cursor from chainStart.
		s.redownloadApproved = s.redownloadApproved[:0]
		s.redownloadCursor = redownloadCursor{
			hash:      s.chainStart.Hash,
			timestamp: s.chainStart.Timestamp,
			height:    s.chainStart.Height,
		}
		s.redownloadChainWork = new(big.Int).Set(s.chainStart.WorkSum)
		s.nextExpectedNBits = s.chainStartNextNBits
		s.phase = PhaseRedownload
		log.Infof("Headers presync transition with peer=%d: "+
			"sufficient work at height=%d, redownloading from height=%d",
			s.peerID, s.currentHeight, s.redownloadCursor.height)
	} else if targetIdx >= 0 {
		hwc := &headers[targetIdx]
		targetHeight := s.nextSpotCheckHeight
		if cert := hwc.BlockCertificate(); cert != nil {
			// Cert-bearing target. When includeCerts is true, the per-header
			// loop already invoked zkpow.VerifyCertificate on this spot-check header, so
			// we trust it.
			if !includeCerts {
				if err := zkpow.VerifyCertificate(&hwc.BlockHeader, cert); err != nil {
					log.Warnf("Headers presync aborted with peer=%d: "+
						"spot-check cert invalid (inline): %v (presync)",
						s.peerID, err)
					s.shouldPunish = true
					return false
				}
			}
			s.armNextSpotCheck(s.currentHeight)
		} else {
			// Cert-less target: issue a getheaders round-trip so the peer
			// must produce the certificate for this exact header.
			h := hwc.BlockHeader
			s.spotCheckHeader = &h
			s.phase = PhasePresyncSpotCheck
			log.Infof("Headers presync spot-check issued with peer=%d: "+
				"height=%d (presync)", s.peerID, targetHeight)
		}
	}
	return true
}

func (s *HeadersSyncState) validateAndProcessSingleHeader(hwc *wire.MsgHeader, includeCerts bool) bool {
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

	if cert := hwc.BlockCertificate(); cert != nil {
		if cert.ProofCommitment() != header.ProofCommitment {
			log.Warnf("Headers presync aborted with peer=%d: "+
				"cert ProofCommitment mismatch at height=%d (presync)",
				s.peerID, nextHeight)
			s.shouldPunish = true
			return false
		}
	}

	if includeCerts {
		cert := hwc.BlockCertificate()
		if cert == nil {
			log.Infof("Headers presync aborted with peer=%d: "+
				"cert missing at height=%d (presync, include_certs=true)",
				s.peerID, nextHeight)
			return false
		}
		if err := zkpow.VerifyCertificate(header, cert); err != nil {
			log.Warnf("Headers presync aborted with peer=%d: "+
				"cert invalid at height=%d (presync, include_certs=true): %v",
				s.peerID, nextHeight, err)
			s.shouldPunish = true
			return false
		}
	}

	headerHash := header.BlockHash()

	bit := s.commitBit(headerHash)
	s.headerCommitments.PushBack(bit)
	if uint64(s.headerCommitments.Len()) > s.maxCommitments {
		log.Infof("Headers presync aborted with peer=%d: "+
			"exceeded max commitments at height=%d (presync)", s.peerID, nextHeight)
		return false
	}

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

// redownloadTipHash returns the hash of the current REDOWNLOAD cursor.
func (s *HeadersSyncState) redownloadTipHash() chainhash.Hash {
	return s.redownloadCursor.hash
}

// redownloadTipTime returns the timestamp of the current REDOWNLOAD cursor.
func (s *HeadersSyncState) redownloadTipTime() int64 {
	return s.redownloadCursor.timestamp
}

func (s *HeadersSyncState) validateAndStoreRedownloadedHeader(hwc *wire.MsgHeader) bool {
	if s.phase != PhaseRedownload {
		return false
	}
	header := &hwc.BlockHeader
	nextHeight := s.redownloadCursor.height + 1

	prevHash := s.redownloadTipHash()
	lastTime := s.redownloadTipTime()
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

	// Advance the cursor to this header.
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

// --- helpers ---

// commitBit computes the 1-bit commitment for a header hash using
// SipHash-2-4 keyed by the per-session hashSalt. The PRF property ensures
// an attacker without the salt cannot predict or bias bits for arbitrary
// headers.
func (s *HeadersSyncState) commitBit(hash chainhash.Hash) bool {
	return siphash.Sum64(hash[:], &s.hashSalt)&1 != 0
}

// maxBlocksSinceStart returns an upper bound on the number of blocks a peer
// could have produced since chainStart, used to cap the PRESYNC commitment
// buffer and scale the certificate-threshold computation.
//
// It is the minimum of two valid upper bounds:
//
//  1. Wall-clock: timestamps must advance by ≥ MinTimestampDeltaSeconds per
//     block, so at most maxSeconds / MinTimestampDeltaSeconds fit.
//
//  2. Work bound: smallest n such that any WTEMA-legal
//     n-1 block chain spanning ≤ maxSeconds must accumulate strictly more
//     work than the peer's budget minimumRequiredWork - chainStart.WorkSum.
func (s *HeadersSyncState) maxBlocksSinceStart() int64 {
	maxSeconds := time.Now().Unix() - s.chainStart.Timestamp + s.chainParams.MaxTimeOffsetMinutes*60
	if maxSeconds <= 0 {
		return 0
	}
	wallBound := maxSeconds / int64(blockchain.MinTimestampDeltaSeconds)
	remainingWork := new(big.Int).Sub(s.minimumRequiredWork, s.chainStart.WorkSum)
	if remainingWork.Sign() <= 0 || wallBound <= 0 {
		return wallBound
	}

	targetSpacing := int64(s.chainParams.TargetTimePerBlock / time.Second)
	halfLife := int64(s.chainParams.WTEMAHalfLife / time.Second)
	if targetSpacing <= int64(blockchain.MinTimestampDeltaSeconds) || halfLife <= 0 {
		return wallBound
	}

	easiestBits := blockchain.CalcEasiestDifficulty(
		s.chainParams, s.chainStart.Bits, time.Duration(maxSeconds)*time.Second)
	wBase, _ := new(big.Float).SetInt(blockchain.CalcWork(easiestBits)).Float64()
	remWork, _ := new(big.Float).SetInt(remainingWork).Float64()
	if !(wBase > 0) || math.IsInf(wBase, 0) || math.IsInf(remWork, 0) {
		// Values outside float64 range; fall back to the wall-clock bound
		// rather than risk a spurious result. In practice wBase and
		// remainingWork are well below 2^1023 for all realistic params.
		return wallBound
	}
	exceeds := func(k int64) bool {
		if k <= 0 {
			return false
		}
		if k > maxSeconds {
			return true
		}
		m := min(max((maxSeconds-k)/(targetSpacing-1), 0), k)
		tail := s.wtemaGrowthSum(k - m)
		return wBase*(float64(m)+tail) > remWork
	}

	lo, hi := int64(1), wallBound+1
	for lo < hi {
		mid := lo + (hi-lo)/2
		if exceeds(mid - 1) {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return min(lo+1, wallBound)
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

func (s *HeadersSyncState) computeCertThreshold() uint32 {
	powLimitBits := s.chainParams.PowLimitBits

	if s.minimumRequiredWork.Cmp(s.chainStart.WorkSum) <= 0 {
		return powLimitBits
	}

	remainingWork := new(big.Int).Sub(s.minimumRequiredWork, s.chainStart.WorkSum)
	n := int64(wire.MaxBlockHeadersPerMsg)

	S := s.wtemaGrowthSum(n)

	maxHeaders := s.maxCommitments
	scaledMax := uint64(float64(maxHeaders) * S)
	if scaledMax == 0 {
		return powLimitBits
	}

	thresholdWork := new(big.Int).Mul(remainingWork, big.NewInt(n))
	thresholdWork.Div(thresholdWork, new(big.Int).SetUint64(scaledMax))
	if thresholdWork.Sign() == 0 {
		return powLimitBits
	}

	// target = (2^256 - 1) / work
	max256 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	thresholdTarget := new(big.Int).Div(max256, thresholdWork)

	powLimit := blockchain.CompactToBig(powLimitBits)
	return blockchain.BigToCompact(blockchain.MinBigInt(thresholdTarget, powLimit))
}

// wtemaGrowthSum returns a + a^2 + ... + a^blocks, where
// a = 1 + targetSpacing/halfLife is the per-block WTEMA work growth factor
// at the 1-second minimum spacing.
func (s *HeadersSyncState) wtemaGrowthSum(blocks int64) float64 {
	targetSpacing := float64(s.chainParams.TargetTimePerBlock / time.Second)
	halfLife := float64(s.chainParams.WTEMAHalfLife / time.Second)
	am1 := targetSpacing / halfLife
	a := 1.0 + am1
	return (math.Pow(a, float64(blocks+1)) - a) / am1
}

// isAtLeastAsHard returns true when nBits a represents at least as much
// difficulty as nBits b (i.e. target_a <= target_b).
func isAtLeastAsHard(a, b uint32) bool {
	targetA := blockchain.CompactToBig(a)
	targetB := blockchain.CompactToBig(b)
	return targetA.Cmp(targetB) <= 0
}
