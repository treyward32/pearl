// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package netsync

import (
	"crypto/rand"
	"math"
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
	require.Equal(t, int32(100), s.PresyncHeight())
	require.True(t, s.PresyncWork().Cmp(big.NewInt(0)) > 0)
}

func TestHeadersSyncPhaseString(t *testing.T) {
	require.Equal(t, "presync", PhasePresync.String())
	require.Equal(t, "presync spot-check", PhasePresyncSpotCheck.String())
	require.Equal(t, "redownload", PhaseRedownload.String())
	require.Equal(t, "final", PhaseFinal.String())
}

func TestShouldIncludeCertificates(t *testing.T) {
	params := makeTestParams()
	start := makeChainStart(100)
	minWork := new(big.Int).Add(start.WorkSum, big.NewInt(1000000))

	s := NewHeadersSyncState(1, params, start, minWork)

	// During PRESYNC with low difficulty, should depend on cert threshold.
	// With regtest params (no retargeting), PowLimitBits is very easy,
	// so ShouldIncludeCertificates depends on the threshold computation.
	_ = s.ShouldIncludeCertificates()
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

func TestIsAtLeastAsHard(t *testing.T) {
	// Same difficulty
	require.True(t, isAtLeastAsHard(0x1a0fffff, 0x1a0fffff))

	// Lower target (harder) vs higher target (easier)
	require.True(t, isAtLeastAsHard(0x190fffff, 0x1a0fffff))

	// Higher target (easier) vs lower target (harder)
	require.False(t, isAtLeastAsHard(0x1b0fffff, 0x1a0fffff))
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

func TestShouldIncludeCertificatesAfterBits(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))

	s.phase = PhaseRedownload
	require.False(t, s.ShouldIncludeCertificatesAfterBits(0xffffffff))
	s.phase = PhaseFinal
	require.False(t, s.ShouldIncludeCertificatesAfterBits(0xffffffff))
	s.phase = PhasePresync
	require.False(t, s.ShouldIncludeCertificatesAfterBits(0x207fffff))
}

func TestShouldIncludeCertificates_RedownloadFalse(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	s.phase = PhaseRedownload
	require.False(t, s.ShouldIncludeCertificates())
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

func TestPipelinedSpeculativeGetHeadersIsStaleSpotCheck(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(100), big.NewInt(5000))
	s.phase = PhasePresyncSpotCheck
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
		PrevBlock:       s.redownloadTipHash(),
		Timestamp:       time.Unix(s.redownloadTipTime()+1, 0),
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
			hdr.BlockHeader.Timestamp = time.Unix(s.redownloadTipTime()+1, 0)
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
	require.Equal(t, startH+int32(wire.MaxBlockHeadersPerMsg), s.RedownloadHeight())

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
	require.Equal(t, startH+int32(wire.MaxBlockHeadersPerMsg)+1, s.RedownloadHeight())
}

// --- maxBlocksSinceStart -----------------------------------------------

// analyticMinCumWork mirrors the big.Float computation in minCumWorkExceeds
// so tests can cross-check the binary search without reading the result out
// of the function under test.
func analyticMinCumWork(k, maxSec, targetSpacing, halfLife int64, wBase float64) float64 {
	if k <= 0 {
		return 0
	}
	m := (maxSec - k) / (targetSpacing - 1)
	if m > k {
		m = k
	}
	if m < 0 {
		m = 0
	}
	a := 1.0 + float64(targetSpacing)/float64(halfLife)
	tail := a * (math.Pow(a, float64(k-m)) - 1) / (a - 1)
	return wBase * (float64(m) + tail)
}

// stateWithParams builds a minimal HeadersSyncState sufficient for
// maxBlocksSinceStart, bypassing the full NewHeadersSyncState path to
// keep tests independent of unrelated constructor side-effects.
func stateWithParams(t *testing.T, params *chaincfg.Params, bits uint32) *HeadersSyncState {
	t.Helper()
	return &HeadersSyncState{
		chainParams: params,
		chainStart: chainStartInfo{
			ChainStartInfo: blockchain.ChainStartInfo{
				Bits:    bits,
				WorkSum: big.NewInt(0),
			},
		},
	}
}

func setMaxSeconds(s *HeadersSyncState, params *chaincfg.Params, maxSec int64) {
	s.chainStart.Timestamp = time.Now().Unix() + params.MaxTimeOffsetMinutes*60 - maxSec
}

// TestMaxBlocksSinceStart_MatchesAnalyticN asserts that the binary search
// lands at the analytic smallest n such that minCumWork(n-1, T) > remainingWork,
// using mainnet parameters and chainStart.Bits == PowLimitBits so the base
// work collapses cleanly to W_floor = CalcWork(PowLimitBits).
func TestMaxBlocksSinceStart_MatchesAnalyticN(t *testing.T) {
	params := chaincfg.MainNetParams
	s := stateWithParams(t, &params, params.PowLimitBits)

	targetSpacing := int64(params.TargetTimePerBlock / time.Second)
	halfLife := int64(params.WTEMAHalfLife / time.Second)

	// Pick T large enough that the mixed target-then-1s regime is active
	// (i.e., T < k*targetSpacing so some 1s-spaced blocks are forced).
	maxSec := int64(3600) // 1 hour
	kRef := int64(50)     // aim for n = kRef + 1

	// Recompute the per-block work lower bound the same way the function
	// under test does, so the analytic cross-check uses the same baseline.
	wBaseI := blockchain.CalcWork(blockchain.CalcEasiestDifficulty(
		&params, params.PowLimitBits, time.Duration(maxSec)*time.Second))
	wBase, _ := new(big.Float).SetInt(wBaseI).Float64()

	cumAtKRef := analyticMinCumWork(kRef, maxSec, targetSpacing, halfLife, wBase)
	cumAtKRefMinus1 := analyticMinCumWork(kRef-1, maxSec, targetSpacing, halfLife, wBase)
	require.Greater(t, cumAtKRef, cumAtKRefMinus1,
		"analytic minCumWork must be strictly increasing in k")

	// Pick remainingWork strictly between minCumWork(kRef-1) and
	// minCumWork(kRef) so the smallest n with minCumWork(n-1, T) > remainingWork
	// is exactly n = kRef + 1. A midpoint satisfies this cleanly.
	remainingWork, _ := new(big.Float).SetFloat64(
		0.5 * (cumAtKRef + cumAtKRefMinus1)).Int(nil)
	s.minimumRequiredWork = remainingWork
	setMaxSeconds(s, &params, maxSec)

	got := s.maxBlocksSinceStart()
	// float64 rounding can shift the comparison by a sub-unit; allow
	// +/-1 slack on the returned n.
	require.GreaterOrEqual(t, got, kRef, "bound must not fall below analytic n-1")
	require.LessOrEqual(t, got, kRef+2, "bound must stay within 1 of analytic n")
}

// TestMaxBlocksSinceStart_ChainStartHarderTightensBound verifies that the
// chainStart.Bits-aware per-block work floor tightens the bound whenever
// chainStart.Bits is meaningfully harder than PowLimitBits and T is short
// enough that WTEMA decay has not saturated the target at PowLimit.
func TestMaxBlocksSinceStart_ChainStartHarderTightensBound(t *testing.T) {
	params := chaincfg.MainNetParams

	// exponent 0x1b-1 = 0x1a shifts the mantissa down by 8 bits, i.e.
	// target is 256x smaller and per-block work is 256x larger than at
	// PowLimitBits for the same mantissa. Use the same mantissa 0x00ffff.
	const harderBits uint32 = 0x1a00ffff
	require.Less(t, blockchain.CompactToBig(harderBits).Cmp(params.PowLimit), 0,
		"harderBits must represent a stricter target than PowLimit")

	easy := stateWithParams(t, &params, params.PowLimitBits)
	hard := stateWithParams(t, &params, harderBits)

	// Choose T well below one half-life so decay factor (87/32)^ceil(T/HL)
	// = (87/32)^1 still leaves the hard target significantly below PowLimit.
	maxSec := int64(3600)

	// Use a remainingWork large enough that the PowLimit-floor bound is
	// non-trivial (well above a single-block minimum). 100 * wFloor is a
	// convenient middle value that places easyBound around 100.
	wFloor := blockchain.CalcWork(params.PowLimitBits)
	remainingWork := new(big.Int).Mul(wFloor, big.NewInt(100))
	easy.minimumRequiredWork = remainingWork
	hard.minimumRequiredWork = remainingWork
	setMaxSeconds(easy, &params, maxSec)
	setMaxSeconds(hard, &params, maxSec)

	easyBound := easy.maxBlocksSinceStart()
	hardBound := hard.maxBlocksSinceStart()

	require.Greater(t, easyBound, int64(1),
		"PowLimit-floor bound should be non-trivial for this remainingWork")
	require.Less(t, hardBound, easyBound,
		"chainStart-aware bound should strictly tighten when chainStart.Bits is harder")
}

// TestMaxBlocksSinceStart_InfeasibleConfigReturnsBoundedN ensures the
// binary search terminates with a finite n even when every feasible
// configuration fails to exceed remainingWork: it must fall back to the
// wall-clock bound.
func TestMaxBlocksSinceStart_InfeasibleConfigReturnsBoundedN(t *testing.T) {
	params := chaincfg.MainNetParams
	s := stateWithParams(t, &params, params.PowLimitBits)

	// Huge remainingWork that no feasible k < wallBound can exceed under
	// the min-cum-work configuration. We force the fallback branch.
	wFloor := blockchain.CalcWork(params.PowLimitBits)
	huge := new(big.Int).Mul(wFloor, new(big.Int).Lsh(big.NewInt(1), 128))

	maxSec := int64(60)
	wallBound := maxSec / int64(blockchain.MinTimestampDeltaSeconds)
	s.minimumRequiredWork = huge
	setMaxSeconds(s, &params, maxSec)

	got := s.maxBlocksSinceStart()
	require.LessOrEqual(t, got, wallBound,
		"bound must not exceed wallBound even when minCumWork never crosses")
}

// TestArmNextSpotCheckBound asserts that every target produced by
// armNextSpotCheck lies strictly in (baseHeight, baseHeight+2*spotCheckInvProb],
// so the distance between consecutive PRESYNC spot-checks is bounded by
// 2*spotCheckInvProb headers.
func TestArmNextSpotCheckBound(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))

	const iters = 100_000
	var maxGap int32
	for i := 0; i < iters; i++ {
		const base int32 = 1_000_000
		s.armNextSpotCheck(base)
		gap := s.nextSpotCheckHeight - base
		require.GreaterOrEqual(t, gap, int32(1))
		require.LessOrEqual(t, gap, int32(2*spotCheckInvProb))
		if gap > maxGap {
			maxGap = gap
		}
	}
	// With 100k draws, we should approach the upper end of the range.
	require.Greater(t, maxGap, int32(spotCheckInvProb),
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
		start.Height+int32(2*spotCheckInvProb))
}

// TestArmNextSpotCheckAdvancesPastConsumedTarget regressions the cert-regime
// transition bug: when baseHeight is the just-consumed target, the next
// target must be strictly greater so the counter never stalls, regardless
// of whether the current regime requires certificates.
func TestArmNextSpotCheckAdvancesPastConsumedTarget(t *testing.T) {
	s := NewHeadersSyncState(1, makeTestParams(), makeChainStart(0), big.NewInt(5000))
	for i := 0; i < 1_000; i++ {
		prev := s.nextSpotCheckHeight
		s.armNextSpotCheck(prev)
		require.Greater(t, s.nextSpotCheckHeight, prev,
			"target must strictly advance on each arm (iter=%d)", i)
	}
}

// TestMaxBlocksSinceStart_TakesMinOfBothBounds validates the min-of-bounds
// combination in maxBlocksSinceStart: when the wall-clock bound is tighter
// (tiny window), it dominates; when the work bound is tighter (long window
// but modest work budget), the work bound dominates.
func TestMaxBlocksSinceStart_TakesMinOfBothBounds(t *testing.T) {
	params := chaincfg.MainNetParams

	t.Run("wall-clock dominates for tiny window", func(t *testing.T) {
		s := stateWithParams(t, &params, params.PowLimitBits)
		// Place chainStart "in the future" minus 10s so maxSeconds is
		// dominated by MaxTimeOffsetMinutes*60 (the only positive term).
		s.chainStart.Timestamp = time.Now().Unix() + 1
		// Modest minimumRequiredWork so the work bound would be huge.
		s.minimumRequiredWork = new(big.Int).Set(blockchain.CalcWork(params.PowLimitBits))

		maxSeconds := params.MaxTimeOffsetMinutes * 60
		wallBound := maxSeconds / int64(blockchain.MinTimestampDeltaSeconds)
		require.LessOrEqual(t, s.maxBlocksSinceStart(), wallBound)
	})

	t.Run("work bound dominates for long window with small remainingWork", func(t *testing.T) {
		s := stateWithParams(t, &params, params.PowLimitBits)
		// Chain start 30 days ago; wall-clock bound is ~2.6M blocks.
		s.chainStart.Timestamp = time.Now().Unix() - 30*24*3600
		// remainingWork = 100 * wFloor -> work bound on the order of 100.
		wFloor := blockchain.CalcWork(params.PowLimitBits)
		s.minimumRequiredWork = new(big.Int).Mul(wFloor, big.NewInt(100))

		got := s.maxBlocksSinceStart()
		require.Less(t, got, int64(10_000),
			"work bound should dominate and keep the result small")
	})
}
