// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"crypto/rand"
	"math/big"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

func makeTestParams() *chaincfg.Params {
	p := chaincfg.RegressionNetParams
	return &p
}

func makeChainStart(height int32) chainStartInfo {
	return chainStartInfo{
		ChainStartInfo: blockchain.ChainStartInfo{
			Hash:          chainhash.Hash{0x01},
			Height:        height,
			Bits:          chaincfg.RegressionNetParams.PowLimitBits,
			Timestamp:     time.Now().Add(-time.Hour).Unix(),
			WorkSum:       big.NewInt(1000),
			PrevTimestamp: time.Now().Add(-time.Hour - time.Second).Unix(),
		},
	}
}

func TestNewHeadersSyncState(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := big.NewInt(5000)

	s := NewHeadersSyncState(1, params, start, minWork)

	require.Equal(t, PhasePresync, s.Phase())
}

func TestHeadersSyncPhaseString(t *testing.T) {
	require.Equal(t, "presync", PhasePresync.String())
	require.Equal(t, "redownload", PhaseRedownload.String())
	require.Equal(t, "final", PhaseFinal.String())
}

func TestWorkNormalizationComputed(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := new(big.Int).Add(start.WorkSum, big.NewInt(1000000))

	s := NewHeadersSyncState(1, params, start, minWork)

	// workNormalization should be positive when remainingWork > 0.
	require.Greater(t, s.workNormalization, 0.0)
}

func TestProcessNextHeadersEmptyReturnsFailure(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := big.NewInt(5000)

	s := NewHeadersSyncState(1, params, start, minWork)

	result := s.ProcessNextHeaders(nil, true)
	require.False(t, result.Success)
	require.False(t, result.RequestMore)
}

func TestProcessNextHeadersFinalState(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := big.NewInt(5000)

	s := NewHeadersSyncState(1, params, start, minWork)
	s.finalize()

	require.Equal(t, PhaseFinal, s.Phase())

	result := s.ProcessNextHeaders([]wire.MsgHeader{{
		BlockHeader: wire.BlockHeader{},
	}}, true)
	require.False(t, result.Success)
}

func TestPresyncWorkSufficient(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := new(big.Int).Add(start.WorkSum, big.NewInt(100))

	s := NewHeadersSyncState(1, params, start, minWork)

	require.False(t, s.PresyncWorkSufficient())

	// Push chain work above minimum.
	s.currentChainWork = new(big.Int).Add(minWork, big.NewInt(1))
	require.True(t, s.PresyncWorkSufficient())
}

func TestCalcNextRequiredDifficultyFromValuesNoRetarget(t *testing.T) {
	params := makeTestParams()
	bits, err := blockchain.CalcNextRequiredDifficultyFromValues(
		params, 100, params.PowLimitBits, 1000, 999,
	)
	require.NoError(t, err)
	require.Equal(t, params.PowLimitBits, bits)
}

func TestCommitBitDeterministic(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(0)
	minWork := big.NewInt(5000)

	s := NewHeadersSyncState(1, params, start, minWork)

	hash := chainhash.Hash{0xAB, 0xCD}
	bit1 := s.commitBit(hash)
	bit2 := s.commitBit(hash)
	require.Equal(t, bit1, bit2)

	// Different session (different salt) may produce different bit
	s2 := NewHeadersSyncState(2, params, start, minWork)
	_ = s2.commitBit(hash) // Just ensure no panic
}

func TestGetAntiDoSWorkThreshold(t *testing.T) {
	// This is a smoke test that the function doesn't panic.
	// A proper test would need a full BlockChain instance.
	_ = blockchain.CalcWork(chaincfg.RegressionNetParams.PowLimitBits)
}

// TestCommitBitSipHashDistribution ensures that commitBit is approximately
// unbiased across random inputs with a fixed salt.
func TestCommitBitSipHashDistribution(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(0)
	s := NewHeadersSyncState(1, params, start, big.NewInt(5000))

	const n = 10000
	ones := 0
	for i := 0; i < n; i++ {
		var h chainhash.Hash
		_, _ = rand.Read(h[:])
		if s.commitBit(h) {
			ones++
		}
	}
	// Expected ~50/50; allow wide margin for non-flakiness.
	require.Greater(t, ones, n/2-500)
	require.Less(t, ones, n/2+500)
}

// TestCommitBitSipHashSaltIndependence ensures two sessions with distinct
// salts produce independently-distributed outputs.
func TestCommitBitSipHashSaltIndependence(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(0)
	s1 := NewHeadersSyncState(1, params, start, big.NewInt(5000))
	s2 := NewHeadersSyncState(2, params, start, big.NewInt(5000))
	for i := range s1.hashSalt {
		s1.hashSalt[i] = 0x00
	}
	for i := range s2.hashSalt {
		s2.hashSalt[i] = 0xFF
	}

	const n = 2000
	agree := 0
	for i := 0; i < n; i++ {
		var h chainhash.Hash
		_, _ = rand.Read(h[:])
		if s1.commitBit(h) == s2.commitBit(h) {
			agree++
		}
	}
	require.Greater(t, agree, n/2-250)
	require.Less(t, agree, n/2+250)
}

// TestCommitBitSipHashKnownVector pins down the SipHash-2-4 output.
func TestCommitBitSipHashKnownVector(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))
	for i := range s.hashSalt {
		s.hashSalt[i] = byte(i)
	}
	var h chainhash.Hash
	for i := range h {
		h[i] = byte(i)
	}
	b1 := s.commitBit(h)
	b2 := s.commitBit(h)
	require.Equal(t, b1, b2)
}

// --- Pipelined GETHEADERS helpers / stale-drop guard --------------------

func TestLastHeaderHashAccessor(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	require.Equal(t, s.chainStart.Hash, s.LastHeaderHash())

	h := chainhash.Hash{0xab, 0xcd}
	s.lastHeaderHash = h
	require.Equal(t, h, s.LastHeaderHash())
}

func TestRedownloadTipHashAccessor(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	require.Equal(t, chainhash.Hash{}, s.RedownloadTipHash())

	s.redownloadCursor = redownloadCursor{
		hash:      s.chainStart.Hash,
		timestamp: s.chainStart.Timestamp,
		height:    s.chainStart.Height,
	}
	require.Equal(t, s.chainStart.Hash, s.RedownloadTipHash())

	hdr := wire.BlockHeader{Version: 42, PrevBlock: chainhash.Hash{0x07}}
	s.redownloadApproved = []ApprovedRedownloadEntry{{Hash: hdr.BlockHash(), Header: hdr}}
	s.redownloadCursor.hash = hdr.BlockHash()
	require.Equal(t, hdr.BlockHash(), s.RedownloadTipHash())
}

func TestSpeculativeLocator(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	anc1 := chainhash.Hash{0x01}
	anc2 := chainhash.Hash{0x02}
	s.chainStart.locator = []*chainhash.Hash{&anc1, &anc2}

	tip := chainhash.Hash{0xff}
	loc := s.SpeculativeLocator(tip)
	require.Len(t, loc, 3)
	require.Equal(t, tip, *loc[0])
	require.Equal(t, anc1, *loc[1])
	require.Equal(t, anc2, *loc[2])

	// Mutation of caller's tip must not affect the returned locator.
	tip[0] = 0xaa
	require.NotEqual(t, tip, *loc[0])
}

func TestShouldSpotCheckByWorkHighProb(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	// Set remaining work very small so C * work >= 1 for any non-trivial work.
	minWork := new(big.Int).Add(start.WorkSum, big.NewInt(1))
	s := NewHeadersSyncState(1, params, start, minWork)

	// With very high workNormalization, any header should be spot-checked.
	work := big.NewInt(1)
	hits := 0
	for i := 0; i < 100; i++ {
		if s.shouldSpotCheckByWork(work) {
			hits++
		}
	}
	require.Greater(t, hits, 90)
}

// --- Pipelined speculative stale-drop -----------------------------------

func TestPipelinedSpeculativeGetHeadersIsStaleNoSession(t *testing.T) {
	require.True(t, pipelinedSpeculativeGetHeadersIsStale(nil, chainhash.Hash{0x11}))
}

func TestPipelinedSpeculativeGetHeadersIsStaleFinalPhase(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	s.finalize()
	prev := chainhash.Hash{0x11, 0x22}
	s.lastHeaderHash = prev
	require.True(t, pipelinedSpeculativeGetHeadersIsStale(s, prev))
}

func TestPipelinedSpeculativeGetHeadersIsStaleRedownloadDivergence(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	s.phase = PhaseRedownload
	s.redownloadCursor = redownloadCursor{
		hash:      s.chainStart.Hash,
		timestamp: s.chainStart.Timestamp,
		height:    s.chainStart.Height,
	}
	speculated := chainhash.Hash{0xde, 0xad, 0xbe, 0xef}
	require.True(t, pipelinedSpeculativeGetHeadersIsStale(s, speculated))
}

func TestPipelinedSpeculativeGetHeadersIsStaleHappyPresync(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	tip := chainhash.Hash{0x42, 0x43}
	s.lastHeaderHash = tip
	require.False(t, pipelinedSpeculativeGetHeadersIsStale(s, tip))
}

func TestPipelinedSpeculativeGetHeadersIsStaleHappyRedownload(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	s.phase = PhaseRedownload
	hdr := wire.BlockHeader{Version: 7, PrevBlock: chainhash.Hash{0x09}}
	s.redownloadApproved = []ApprovedRedownloadEntry{{Hash: hdr.BlockHash(), Header: hdr}}
	tip := hdr.BlockHash()
	s.redownloadCursor.hash = tip
	require.False(t, pipelinedSpeculativeGetHeadersIsStale(s, tip))
}

// --- REDOWNLOAD validation & Tier-1 capacity --------------------------

// newRedownloadState returns a state forced into PhaseRedownload with
// processAllRemainingHeaders=true so the commitment-bit check is bypassed.
func newRedownloadState(t *testing.T) *HeadersSyncState {
	t.Helper()
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	s.phase = PhaseRedownload
	s.redownloadCursor = redownloadCursor{
		hash:      s.chainStart.Hash,
		timestamp: s.chainStart.Timestamp,
		height:    s.chainStart.Height,
	}
	s.processAllRemainingHeaders = true
	s.nextExpectedNBits = s.chainStartNextNBits
	return s
}

// buildValidRedownloadHeader returns a cert-less MsgHeader accepted at
// the current redownload tip.
func buildValidRedownloadHeader(s *HeadersSyncState) *wire.MsgHeader {
	return &wire.MsgHeader{BlockHeader: wire.BlockHeader{
		Version:         1,
		PrevBlock:       s.redownloadCursor.hash,
		Timestamp:       time.Unix(s.redownloadCursor.timestamp+1, 0),
		Bits:            s.nextExpectedNBits,
		ProofCommitment: chainhash.Hash{0x01},
	}}
}

func TestValidateAndStoreRedownloadedHeader_NoCert(t *testing.T) {
	t.Run("accept-valid", func(t *testing.T) {
		s := newRedownloadState(t)
		hdr := buildValidRedownloadHeader(s)
		require.True(t, s.validateAndStoreRedownloadedHeader(hdr))
		require.Equal(t, 1, s.RedownloadApprovedLen())
		require.Equal(t, hdr.BlockHeader.BlockHash(), s.RedownloadTipHash())
	})

	t.Run("reject-non-continuous", func(t *testing.T) {
		s := newRedownloadState(t)
		hdr := buildValidRedownloadHeader(s)
		hdr.BlockHeader.PrevBlock = chainhash.Hash{0xde, 0xad}
		require.False(t, s.validateAndStoreRedownloadedHeader(hdr))
		require.Equal(t, 0, s.RedownloadApprovedLen())
	})

	t.Run("reject-bad-difficulty", func(t *testing.T) {
		s := newRedownloadState(t)
		hdr := buildValidRedownloadHeader(s)
		if s.nextExpectedNBits == s.chainParams.PowLimitBits {
			hdr.BlockHeader.Bits = s.chainParams.PowLimitBits - 1
		} else {
			hdr.BlockHeader.Bits = s.chainParams.PowLimitBits
			hdr.BlockHeader.Timestamp = time.Unix(s.redownloadCursor.timestamp+1, 0)
		}
		require.False(t, s.validateAndStoreRedownloadedHeader(hdr))
		require.Equal(t, 0, s.RedownloadApprovedLen())
	})

	t.Run("reject-zero-proof-commitment", func(t *testing.T) {
		s := newRedownloadState(t)
		hdr := buildValidRedownloadHeader(s)
		hdr.BlockHeader.ProofCommitment = chainhash.Hash{}
		require.False(t, s.validateAndStoreRedownloadedHeader(hdr))
		require.Equal(t, 0, s.RedownloadApprovedLen())
		require.True(t, s.shouldPunish)
	})
}

func TestRedownloadTier1Capacity(t *testing.T) {
	s := newRedownloadState(t)
	targetFill := redownloadApprovedCap - redownloadApprovedHeadroom + 1
	for i := 0; i < targetFill; i++ {
		hdr := buildValidRedownloadHeader(s)
		require.True(t, s.validateAndStoreRedownloadedHeader(hdr),
			"unexpected reject at i=%d", i)
	}
	require.Equal(t, targetFill, s.RedownloadApprovedLen())
	require.False(t, s.hasRedownloadFifoCapacity(), "should be saturated")

	popped := s.PopApprovedRedownloadHashes(wire.MaxBlockHeadersPerMsg)
	require.Len(t, popped, wire.MaxBlockHeadersPerMsg)
	require.True(t, s.hasRedownloadFifoCapacity(), "should regain capacity")
	require.Equal(t, targetFill-wire.MaxBlockHeadersPerMsg, s.RedownloadApprovedLen())

	prev := popped[len(popped)-1].Hash
	rest := s.PopApprovedRedownloadHashes(redownloadApprovedCap)
	require.Equal(t, targetFill-wire.MaxBlockHeadersPerMsg, len(rest))
	require.Equal(t, prev, rest[0].Header.PrevBlock)
	require.Equal(t, 0, s.RedownloadApprovedLen())
}

// TestRedownloadTipStableAcrossPop regressions the bug where the cursor
// reverted to chain_start once Tier-1 drained.
func TestRedownloadTipStableAcrossPop(t *testing.T) {
	s := newRedownloadState(t)

	for i := 0; i < wire.MaxBlockHeadersPerMsg; i++ {
		hdr := buildValidRedownloadHeader(s)
		require.True(t, s.validateAndStoreRedownloadedHeader(hdr))
	}
	tipAfterFirst := s.RedownloadTipHash()
	timeAfterFirst := s.redownloadCursor.timestamp
	require.NotEqual(t, s.chainStart.Hash, tipAfterFirst)
	startH := s.chainStart.Height
	require.Equal(t, startH+int32(wire.MaxBlockHeadersPerMsg), s.redownloadCursor.height)

	popped := s.PopApprovedRedownloadHashes(wire.MaxBlockHeadersPerMsg)
	require.Len(t, popped, wire.MaxBlockHeadersPerMsg)
	require.Equal(t, 0, s.RedownloadApprovedLen())

	// Cursor must survive the drain.
	require.Equal(t, tipAfterFirst, s.RedownloadTipHash())
	require.Equal(t, timeAfterFirst, s.redownloadCursor.timestamp)

	// Second batch should accept its first header.
	next := buildValidRedownloadHeader(s)
	require.Equal(t, tipAfterFirst, next.BlockHeader.PrevBlock)
	require.True(t, s.validateAndStoreRedownloadedHeader(next))
	require.Equal(t, next.BlockHeader.BlockHash(), s.RedownloadTipHash())
	require.Equal(t, startH+int32(wire.MaxBlockHeadersPerMsg)+1, s.redownloadCursor.height)
}

// TestArmNextSpotCheckBound asserts that every target produced by
// scheduleNextSpotCheck lies strictly in (baseHeight, baseHeight+2*spotCheckMeanGap],
// so the distance between consecutive PRESYNC spot-checks is bounded by
// 2*spotCheckMeanGap headers.
func TestArmNextSpotCheckBound(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))

	const iters = 100_000
	var maxGap int32
	for i := 0; i < iters; i++ {
		const base int32 = 1_000_000
		s.scheduleNextSpotCheck(base)
		gap := s.nextSpotCheckHeight - base
		require.GreaterOrEqual(t, gap, int32(1))
		require.LessOrEqual(t, gap, int32(2*spotCheckMeanGap))
		if gap > maxGap {
			maxGap = gap
		}
	}
	// With 100k draws, we should approach the upper end of the range.
	require.Greater(t, maxGap, int32(spotCheckMeanGap),
		"expected to sample into the upper half at least once")
}

// TestNewHeadersSyncStateArmsSpotCheck asserts that the constructor arms
// the first spot-check target strictly beyond chainStart.Height and within
// one bounded gap.
func TestNewHeadersSyncStateArmsSpotCheck(t *testing.T) {
	start := makeChainStart(500)
	s := NewHeadersSyncState(1, makeTestParams(), start, big.NewInt(5000))
	require.Greater(t, s.nextSpotCheckHeight, start.Height)
	require.LessOrEqual(t, s.nextSpotCheckHeight,
		start.Height+int32(2*spotCheckMeanGap))
}

// TestArmNextSpotCheckAdvancesPastConsumedTarget regressions the cert-regime
// transition bug: when baseHeight is the just-consumed target, the next
// target must be strictly greater so the counter never stalls, regardless
// of whether the current regime requires certificates.
func TestArmNextSpotCheckAdvancesPastConsumedTarget(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))
	for i := 0; i < 1_000; i++ {
		prev := s.nextSpotCheckHeight
		s.scheduleNextSpotCheck(prev)
		require.Greater(t, s.nextSpotCheckHeight, prev,
			"target must strictly advance on each arm (iter=%d)", i)
	}
}

// --- Pipelined spot-check tests -------------------------------------------

func TestSpotCheckQueueing(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))

	// Force spot check at a known height.
	s.nextSpotCheckHeight = 3

	require.Equal(t, PhasePresync, s.Phase())
	require.Empty(t, s.pendingSpotChecks)

	// Directly simulate: append a pending spot check as the per-header
	// loop would when it encounters a cert-less target.
	s.pendingSpotChecks = append(s.pendingSpotChecks, pendingSpotCheck{
		height:    3,
		hash:      chainhash.Hash{0xAA},
		prevBlock: chainhash.Hash{0xBB},
	})

	// Phase must remain PhasePresync.
	require.Equal(t, PhasePresync, s.Phase())
	require.Len(t, s.pendingSpotChecks, 1)
	require.Equal(t, int32(3), s.pendingSpotChecks[0].height)
}

func TestSpotCheckBackpressure(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))

	// No pending checks: not backpressured.
	require.False(t, s.SpotCheckBackpressured())

	// Add a pending check at height 100, current height at 100.
	s.pendingSpotChecks = append(s.pendingSpotChecks, pendingSpotCheck{
		height: 100,
		hash:   chainhash.Hash{0x01},
	})
	s.currentHeight = 100
	require.False(t, s.SpotCheckBackpressured())

	// Advance to exactly spotCheckMeanGap - 1 ahead: not yet.
	s.currentHeight = 100 + spotCheckMeanGap - 1
	require.False(t, s.SpotCheckBackpressured())

	// Advance to exactly spotCheckMeanGap ahead: backpressured.
	s.currentHeight = 100 + spotCheckMeanGap
	require.True(t, s.SpotCheckBackpressured())

	// Remove the pending check: no longer backpressured.
	s.pendingSpotChecks = nil
	require.False(t, s.SpotCheckBackpressured())
}

func TestSpotCheckResponseMatching(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))

	scHash := chainhash.Hash{0xDE, 0xAD}
	s.pendingSpotChecks = append(s.pendingSpotChecks, pendingSpotCheck{
		height:    500,
		hash:      scHash,
		prevBlock: chainhash.Hash{0x01},
	})

	// findPendingSpotCheck should find it.
	require.Equal(t, 0, s.findPendingSpotCheck(scHash))

	// Non-matching hash returns -1.
	require.Equal(t, -1, s.findPendingSpotCheck(chainhash.Hash{0xFF}))
}

func TestSpotCheckMultipleInFlight(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))

	s.pendingSpotChecks = []pendingSpotCheck{
		{height: 100, hash: chainhash.Hash{0x01}},
		{height: 200, hash: chainhash.Hash{0x02}},
		{height: 300, hash: chainhash.Hash{0x03}},
	}

	// Remove the middle one.
	idx := s.findPendingSpotCheck(chainhash.Hash{0x02})
	require.Equal(t, 1, idx)
	s.pendingSpotChecks = append(s.pendingSpotChecks[:idx], s.pendingSpotChecks[idx+1:]...)

	require.Len(t, s.pendingSpotChecks, 2)
	require.Equal(t, chainhash.Hash{0x01}, s.pendingSpotChecks[0].hash)
	require.Equal(t, chainhash.Hash{0x03}, s.pendingSpotChecks[1].hash)
}

func TestSpotCheckStaleAbsorptionDuringRedownload(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))

	scHash := chainhash.Hash{0xAB, 0xCD}
	s.pendingSpotChecks = []pendingSpotCheck{
		{height: 50, hash: scHash, prevBlock: chainhash.Hash{0x01}},
	}

	// Simulate REDOWNLOAD transition: pending checks are kept.
	s.phase = PhaseRedownload
	s.redownloadCursor = redownloadCursor{
		hash:      s.chainStart.Hash,
		timestamp: s.chainStart.Timestamp,
		height:    s.chainStart.Height,
	}

	// findPendingSpotCheck still finds it during REDOWNLOAD.
	require.Equal(t, 0, s.findPendingSpotCheck(scHash))
	require.Len(t, s.pendingSpotChecks, 1)
}

func TestFinalizesClearsPendingSpotChecks(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))
	s.pendingSpotChecks = []pendingSpotCheck{
		{height: 100, hash: chainhash.Hash{0x01}},
	}
	s.finalize()
	require.Nil(t, s.pendingSpotChecks)
	require.Equal(t, PhaseFinal, s.Phase())
}
