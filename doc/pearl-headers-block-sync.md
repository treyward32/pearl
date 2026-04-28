# Pearl Headers / Block Sync

## Introduction

Pearl replaces Bitcoin's nonce-based proof of work with a ZK proof-of-work certificate (~60 KB per block). This changes header synchronisation in three ways:

1. **PoW is not self-contained in the header.** Verification requires an associated certificate.
2. **PoW verification is expensive.** Verifying every certificate unconditionally during initial sync exposes the receiver to CPU-DoS from malicious peers streaming long low-difficulty chains.
3. **Certificates are large.** Sending certificates for every header in every (possibly malicious) batch would substantially increase bandwidth cost.

Pearl therefore keeps Bitcoin's two-pass anti-DoS structure but changes what is checked in each pass:

> **Note:** The primary goal of this two-phase syncing is to protect against memory DoS (storing too many unverified blocks), and *not* against futile block verification time DoS.

- **PRESYNC** performs cheap structural checks, stores dense commitments, accumulates provisional chain work, and verifies certificates only selectively (above-threshold batches and random spot-checks).
- **REDOWNLOAD** replays the same chain *without certificates in HEADERS*, runs cheap contextual checks against PRESYNC commitments, and defers ZK certificate verification to block-arrival time (each certificate travels inside the full-block payload, so it is transferred exactly once).

**Invariants:**

1. Each header contains a binding commitment (`ProofCommitment`) to its certificate.
2. PRESYNC work is provisional -- used only for anti-DoS gating.
3. No header is permanently accepted without certificate verification.
4. REDOWNLOAD upgrades a provisional chain into an accepted chain.

---

## Wire Protocol

Pearl reuses Bitcoin's `GETHEADERS` / `HEADERS` flow with one extension.

### GETHEADERS

Fields: `locator`, `hashStop`, `IncludeCertificates` (bool, serialised as 1 byte).

- `IncludeCertificates=false` — return headers only.
- `IncludeCertificates=true` (default) — return headers with the certificate for every returned header.

Certificate inclusion is batch-level, not per-header.

### HEADERS

Returns up to `MAX_HEADERS_RESULTS` (100) consecutive headers from the first locator match toward `hashStop` (or toward the peer's tip if `hashStop` is zero). When `IncludeCertificates=true`, each header is accompanied by its certificate.

### Inconsistent-cert ban rule

If a single HEADERS message contains a mix of headers with and without certificates, the peer is banned immediately. By construction, either all headers in a message carry certs or none do.

---

## Common Header Checks

The following structural checks apply in both PRESYNC and REDOWNLOAD. The implementation shares them via a single validation path.

1. **Continuity** — `header.PrevBlock` must match the hash of the preceding header (or the phase-specific tip for the first header in a batch).
2. **Timestamp monotonicity** — `header.Timestamp >= prev.Timestamp + MIN_BLOCK_INTERVAL` (1 second).
3. **Difficulty transition** — `header.Bits` must equal the value produced by the WTEMA rule (`CalcNextRequiredDifficulty`). Exception: on testnet with `ReduceMinDifficulty`, `PowLimitBits` is accepted when the block time exceeds `MinDiffReductionTime` past the previous timestamp.
4. **ProofCommitment well-formedness** — `header.ProofCommitment` must be non-zero. A zero value is punishable.
5. **Certificate / header consistency** — when a certificate is present alongside the header, `cert.ProofCommitment() == header.ProofCommitment` must hold. A mismatch is punishable.

---

## Anti-DoS Work Threshold

Before routing incoming headers, the node computes a work threshold:

```
threshold = max(tip.workSum - antiDoSBufferBlocks * CalcWork(tip.Bits),
                MinimumChainWork)
```

where `antiDoSBufferBlocks = 100`.

- **Above threshold**: headers are accepted directly via `AcceptBlockHeader` (requires certificates). If certificates are absent, the node re-requests the same range with `IncludeCertificates=true`.
- **Below threshold**: the node initiates a presync session for that peer.

Presync can be triggered from two paths:

- **Headers path** (`handleHeadersMsg`): the first batch's claimed work (fork-point work + batch work) falls below the threshold.
- **Block path** (`handleBlockMsg`): a block arrives whose parent chain has insufficient work. The block is discarded and presync is started from the parent.

---

## Pre-validation

Before any presync or direct-acceptance routing, every incoming HEADERS message undergoes pre-validation:

1. **nBits format** — each header's `Bits` must encode a valid target within `PowLimit`. Invalid nBits disconnects the peer.
2. **Intra-batch continuity** — each header's `PrevBlock` must equal the preceding header's hash. Violation disconnects the peer.

These checks apply universally, regardless of whether the peer has an active presync session.

---

## Phase 1: PRESYNC

### Goal

Determine whether a peer can demonstrate a chain with at least `minimum_required_work`, without inserting headers into the permanent block index.

### Entry conditions

A presync session is created when:

- the incoming headers' total work (from the fork point) is below the anti-DoS threshold, AND
- the batch is a full message (`MAX_HEADERS_RESULTS` headers). A partial low-work batch is silently ignored.

### Per-header processing

For each header, the receiver:

1. Performs the common header checks (see above).
2. Stores a **1-bit commitment**: `salted_hash(header.BlockHash()) & 1`. The salt is a random 16-byte key generated once per session. Every header gets one bit (denser than Bitcoin; Pearl defers many certificate checks to REDOWNLOAD, so PRESYNC needs stronger replay binding).
3. Accumulates **provisional chain work** from `header.Bits`.

If the commitment deque exceeds `maxCommitments` (= `maxBlocksSinceStart`, derived from wall-clock time and `MIN_BLOCK_INTERVAL`), the presync is aborted.

### Certificate policy

Certificate verification during PRESYNC is batch-level. For each outgoing `GETHEADERS`, the receiver decides:

```
ShouldIncludeCertificates():
    switch phase:
    case PRESYNC:             return isAtLeastAsHard(lastHeader.Bits, certThresholdNBits)
    case PRESYNC_SPOT_CHECK:  return true
    case REDOWNLOAD:          return false   // cert travels inside the block
    case FINAL:               return false   // no getheaders issued
```

- **Above-threshold PRESYNC batches**: `IncludeCertificates=true`. Every certificate in the batch is verified.
- **Below-threshold PRESYNC batches**: `IncludeCertificates=false`. No ordinary certificate verification, but random spot-checks may apply.
- **PRESYNC_SPOT_CHECK**: `IncludeCertificates=true` for the one header being re-requested.
- **REDOWNLOAD**: `IncludeCertificates=false`. The certificate is bundled into the full-block payload we will request via `getdata`, so piggybacking it on the HEADERS response would double the bytes for no benefit.

The provisional work accumulated during PRESYNC is not fully validated: above-threshold work is backed by certificate verification; below-threshold work is not yet certificate-verified and is used only to decide whether the chain is worth redownloading.

### Certificate threshold

A fixed threshold `certThresholdNBits` is computed once when the session is created, using the chain-start state:

```
maxRemainingBlocks = (now - startTime + TIMESTAMP_WINDOW) / MIN_BLOCK_INTERVAL
a = 1 + targetSpacing / wtemaHalfLife
S = (a^(N+1) - a) / (a - 1)        where N = MAX_HEADERS_RESULTS
thresholdWork = remainingWork / maxRemainingBlocks * N / S
certThresholdNBits = min(workToTarget(thresholdWork), powLimitBits)
```

Where:

- `remainingWork` = work needed to reach `minimum_required_work` from chain_start.
- `maxRemainingBlocks` = upper bound on how many blocks the peer's chain could contain since chain_start, given wall-clock time and `TIMESTAMP_WINDOW`.
- `N = MAX_HEADERS_RESULTS`.

The threshold ensures the high-difficulty region, which contributes most of the work toward `minimum_required_work`, is fully certificate-verified during PRESYNC. It remains constant for the entire session.

### Random spot-checks

For each below-threshold batch, the receiver draws a random index:

```
spotCheckIdx = rand.Int63n(SPOT_CHECK_INV_PROB)
```

All headers in the batch are processed normally. After the batch:

- If `provisional_work >= minimum_required_work`, transition to REDOWNLOAD (takes priority).
- Otherwise, if `spotCheckIdx < len(headers)`, the header at that index is selected for spot-checking.

When a header is selected:

1. **Certificate present in batch** (batch happened to include certs): verify `CheckProofOfWork(header, cert)`. Invalid certificate is punishable. Otherwise continue normally.
2. **Certificate not present**: save the header, transition to `PRESYNC_SPOT_CHECK`. Issue a one-header `GETHEADERS` positioned so the selected header is the first returned, with `IncludeCertificates=true`. On reply:
   - Hash mismatch → abort without punishment (peer may have answered a different query).
   - Certificate missing → abort without punishment.
   - Invalid certificate → punishable.
   - Valid → return to PRESYNC, resume from the stored presync tip.

---

## Phase 2: REDOWNLOAD

### Goal

Replay the same chain *without* fetching certificates in HEADERS, run the cheap contextual checks on every header (continuity, difficulty, timestamp monotonicity, non-zero `ProofCommitment`, and the salted commitment-bit cross-check), and defer ZK certificate verification to block-arrival time. Every block already carries its certificate, so the cert traverses the wire exactly once instead of twice.

### Two-tier buffer

REDOWNLOAD maintains two bounded in-memory buffers for the session:

| Tier | Location | Contents | Bound | Back-pressure |
|---|---|---|---|---|
| Tier-1 | `HeadersSyncState.redownloadApproved` | Approved cert-less `ApprovedRedownloadEntry{Hash, Header}` awaiting block-body fetch (pure pending-emission queue; the REDOWNLOAD tip used for locator construction and continuity checks lives on a separate cursor, see below) | `redownloadApprovedCap` = **500** | On saturation `ProcessNextHeaders` withholds `RequestMore`, and the state machine stops asking the peer for more HEADERS until the `SyncManager` drains the FIFO. |
| Tier-2 | `peerSyncState.redownloadExpected` + `.redownloadPendingBlocks` | In-order FIFO of entries for which `getdata` has been sent (possibly with the block already arrived and buffered pending in-order acceptance) | `redownloadPendingCap` = **100** | When full, `driveRedownloadGetdata` stops popping from Tier-1; Tier-1 fills up; which in turn stops the next getheaders via the Tier-1 mechanism above. |

Because Tier-2 back-pressure transparently propagates up through Tier-1 to the getheaders loop, the state machine never needs to block inside `ProcessNextHeaders` and a slow block-body pipeline cannot force us to buffer arbitrary amounts of header data.

### REDOWNLOAD tip cursor

The REDOWNLOAD tip is tracked by a dedicated cursor `HeadersSyncState.redownloadCursor` holding `{hash, timestamp, height}`. The companion `nextExpectedNBits` field is re-seeded at the PRESYNC→REDOWNLOAD transition and advanced in lockstep. The cursor is:

- **Initialized** from `chainStart` at the PRESYNC→REDOWNLOAD transition inside `validateAndStoreCommitments`.
- **Advanced** exclusively inside `validateAndStoreRedownloadedHeader`, after a header has passed all cheap checks and been appended to Tier-1.
- **Never reset by Tier-1 drain.** `PopApprovedRedownloadHashes` moves entries into Tier-2 but does not touch the cursor.

The cursor is the single source of truth for:

- the prev-block reference used by the continuity check on the next incoming header,
- the timestamp used by the monotonicity check,
- the locator head produced by `NextHeadersRequestLocator` and `RedownloadTipHash`,
- the phase-aware height reported by `CurrentHeight` (used by operational logs so the REDOWNLOAD replay position is visible instead of the frozen PRESYNC endpoint).

Decoupling the cursor from Tier-1 is required because speculative GETHEADERS and the continuity check must keep working while Tier-1 is being drained into Tier-2. Deriving the tip from the last element of Tier-1 snapped it back to `chainStart` as soon as the FIFO emptied, breaking both the next-batch continuity check and the locator on the next speculative request.

### Per-batch timeline

1. **Request** — `GETHEADERS` is issued with `IncludeCertificates=false`. REDOWNLOAD is the only phase that explicitly turns the cert flag off (`ShouldIncludeCertificates` and `ShouldIncludeCertificatesAfterBits` both return `false` for `PhaseRedownload`). The locator is `[cursor.hash, ...chainStart.locator]` where `cursor.hash` is the REDOWNLOAD tip cursor described above (the hash of the last header appended to Tier-1, or `chainStart.Hash` before the first append). The cursor is stable across Tier-1 drain.
2. **Receive + cheap validate** — for each header `validateAndStoreRedownloadedHeader` performs: the common header checks (continuity, timestamp monotonicity, difficulty); non-zero `ProofCommitment`; and the 1-bit salted-commitment cross-check against the PRESYNC deque while within the PRESYNC commitment window (see *Commitment bit primitive* below). On success the `{hash, header}` pair is appended to Tier-1 and work is accumulated. **No ZK verification runs at this step.** Certificates never arrive in HEADERS during REDOWNLOAD.
3. **Drain Tier-1 into `getdata`** — `driveRedownloadGetdata` pops up to `redownloadPendingCap - len(redownloadExpected)` entries from the Tier-1 head, tracks them in Tier-2, and emits one `GETDATA(InvTypeWitnessBlock)` batch. Popped entries retain their approved `BlockHeader` so the arriving block can be cross-checked byte-for-byte against it.
4. **Next `GETHEADERS`** — after clearing the per-peer rate limit, `isContinuationOfLowWorkHeadersSync` issues the next REDOWNLOAD `GETHEADERS` (cert-less) via the existing pipelining path. If Tier-1 was saturated, `RequestMore` is suppressed; a later block-arrival drain re-triggers the follow-up request via `maybeTriggerRedownloadGetHeaders`.
5. **Block arrives** — `handleRedownloadBlockArrival` (invoked from the top of `handleBlockMsg`) locates the matching Tier-2 entry, verifies that the arriving block's header hashes to the approved hash, stashes the block in `redownloadPendingBlocks`, and drains the Tier-2 head while successive blocks are present. Each drain invokes `chain.ProcessBlock(block, BFNone)`, which in turn runs `checkProofOfWork` → `zkpow.VerifyCertificate(header, cert)`: the cert travels inside the block and is verified here for the first and only time. On success the block is accepted into the chain index and the main chain advances.

### DoS argument

Replacing the cert-carrying REDOWNLOAD HEADERS with cert-less HEADERS + getdata-for-body is safe because:

- **Work still pays to get past PRESYNC.** A peer must still deliver a chain whose claimed cumulative work hits `minimum_required_work`, with above-threshold batches cert-verified inline and 1-in-1000 spot checks binding the below-threshold region.
- **Salted commitment bits still bind the chain.** Any substitution of a REDOWNLOAD header is caught by the `hash → SipHash-2-4(salt, hash) & 1` cross-check against the PRESYNC deque, as long as we are within the commitment window. Outside the window (`processAllRemainingHeaders`), the accumulated work requirement has already been met, so any further header would only add to that accumulation.
- **Cert verification is block-gated, not header-gated.** No REDOWNLOAD header enters the block index until its block has arrived and `zkpow.VerifyCertificate` has succeeded on the cert carried in the block payload. A peer who withholds blocks only stalls the single peer session; it cannot cause us to accept bogus headers.
- **Header-bytes-equal on arrival.** The arriving block's `BlockHeader` must exactly match the approved Tier-1 header (we compare `BlockHash()`, which covers all serialisable header fields). Any discrepancy is a punishable peer violation — the peer tried to bait-and-switch the approval with a different header whose cert they actually own.
- **Bounded memory.** Tier-1 ≤ 500 headers × ~80 B ≈ 40 KB per session. Tier-2 ≤ 100 blocks × header+cert ≈ ~6 MB per peer. A malicious peer cannot force us past these bounds; if they stall, we back-pressure; if they violate, we disconnect.

### Commitment bit primitive

`commitBit(hash)` is a 1-bit keyed pseudo-random function bound to the session:

```
commitBit(hash) = SipHash-2-4(hashSalt, hash)[0] & 1
```

`hashSalt` is a 128-bit random key drawn from `crypto/rand` once per session (`NewHeadersSyncState`). Because SipHash-2-4 is a cryptographic PRF, an attacker who does not know `hashSalt` cannot predict or bias commitment bits for arbitrary headers, which is the structural property both the random spot-check and the REDOWNLOAD commitment cross-check rely on. The single bit is enough: any substituted header succeeds with ≤ 50% probability per position, so any substitution of length > 80 has probability < 2⁻⁸⁰.

### Completion

REDOWNLOAD's headers phase is considered complete when the peer answers a REDOWNLOAD `GETHEADERS` with a short batch (< `MAX_HEADERS_RESULTS`) after `processAllRemainingHeaders` has already triggered (i.e. after we have already accumulated ≥ `minimum_required_work`). The state machine sets an internal `redownloadShortBatchSeen` flag and continues emitting any remaining approved entries through `PopApprovedRedownloadHashes`. When both conditions hold — no more headers expected (`redownloadShortBatchSeen == true`) *and* Tier-1/Tier-2 are drained — the `SyncManager` clears `state.headersSyncState` and the session ends.

Block-body processing continues independently through the ordinary `handleBlockMsg` path once the state machine is dismissed, because every REDOWNLOAD-approved header has already been entered into the block index through `ProcessBlock` by the time we reach this point.

An incomplete HEADERS response before `processAllRemainingHeaders` is still a non-punishable abort: the peer stopped short of demonstrating the advertised work, and the session is torn down with the peer marked low-quality.

---

## Phase 3: Block Download

Above-threshold HEADERS batches and post-presync tip extensions continue to use the ordinary block-download path:

- For each peer the node tracks the best announced block and the last shared full block.
- Missing full blocks are requested via `GETDATA(MSG_BLOCK)`.
- When the node is within `antiDoSBufferBlocks` of the tip, blocks may be fetched directly upon header acceptance (skipping the header queue).

During an active REDOWNLOAD session, the ordinary path is bypassed: block bodies for REDOWNLOAD-approved headers flow exclusively through the Tier-1/Tier-2 pipeline described above. This preserves the in-order acceptance invariant (`AcceptBlockHeader` is never called on a header whose predecessor is not yet in the chain index) while allowing block download to overlap with header validation.

Each full block contains the header, the certificate, and the transactions. `AcceptBlock` verifies the certificate in the full-block validation path, and REDOWNLOAD relies on this single cert verification — no additional ZK verification runs during REDOWNLOAD header processing.

---

## Persistence and Memory Safety

### What gets written to disk

The block index and chain DB are **never written** during PRESYNC or the REDOWNLOAD headers phase. Headers processed in either phase live exclusively in `HeadersSyncState` memory (commitment deque, Tier-1). `AcceptBlockHeader` is not called for any presync-routed header. The only persistent write that can result from a presync session is `chain.ProcessBlock` on a full block during REDOWNLOAD block arrival — which only succeeds after:

1. The arriving block's header matches the Tier-1 approved header byte-for-byte.
2. `zkpow.VerifyCertificate` passes on the certificate carried in the block.
3. Full block validation succeeds.

A peer that fails presync or aborts REDOWNLOAD early causes **zero persistent state** beyond any blocks already accepted by the above path (which are fully verified).

### Session memory is bounded and freed on abort

| Object | Location | Bound |
|---|---|---|
| Commitment deque | `HeadersSyncState.headerCommitments` | ≤ `maxCommitments` bits (≈ 252 K bits ≈ 32 KB) |
| Tier-1 | `HeadersSyncState.redownloadApproved` | ≤ `redownloadApprovedCap` = 500 entries |
| Tier-2 | `peerSyncState.redownloadExpected` + `redownloadPendingBlocks` | ≤ `redownloadPendingCap` = 100 entries |

On any PRESYNC failure, `finalize()` clears the commitment deque and Tier-1 immediately inside `ProcessNextHeaders`. On REDOWNLOAD abort, `resetRedownloadTier2` clears Tier-2 and the session pointer is nulled, releasing Tier-1 with it.

### getdata batching and commitment coverage

`driveRedownloadGetdata` issues **one batched `GETDATA`** per call containing up to `min(redownloadPendingCap - len(tier2), EligibleForGetdata())` block inv entries (typically up to 100 blocks per call). Two conditions must hold before an entry can be popped from Tier-1 to Tier-2:

1. **Tier-2 has free capacity** (`redownloadPendingCap - len(tier2) > 0`).
2. **Depth guard** (`redownloadGetdataDepth` = 100): the entry must have at least 100 more commitment-checked headers on top of it in Tier-1. Only the leading `len(tier1) - 100` entries are eligible at any moment. When the headers phase is complete (`redownloadShortBatchSeen = true`), the depth guard is lifted and all remaining entries become immediately eligible.

All headers popped to Tier-2 while within the PRESYNC commitment window have already had their 1-bit salted commitment cross-checked against the PRESYNC deque (one bit per header in sequence). After `processAllRemainingHeaders = true` (cumulative redownload work ≥ `minimum_required_work`), the per-header bit-check is skipped — safe because the work requirement is already satisfied.

### Presync entry gating

Presync is initiated only when an **unrequested** block arrives whose parent is already a known chain entry (`LookupChainStartInfo` succeeds) and whose parent's `WorkSum` is below `getAntiDoSWorkThreshold()`. The triggering block itself is discarded without being passed to `ProcessBlock`. Blocks with an unknown parent are silently ignored (orphan handling) and cannot start a presync session.

---

## Peer Quality and Tip Extension

### shouldDownloadBlocks

A chain is worth full-block download when:

```
csi.WorkSum + CalcWork(nextHypotheticalBits) >= bestTip.workSum
```

where `nextHypotheticalBits` is derived from the chain's last two blocks via WTEMA.

### Quality counter

Each peer has an integer quality counter, starting at `peerQualityThreshold` (5, i.e. low-quality).

- **Reset to 0** (high-quality) when the peer sends a block that becomes the new chain tip.
- **Incremented** when the peer sends a block or header that is not near-tip.
- **Set above threshold** (low-quality) when a presync session with the peer fails silently.
- If a peer sends a tip/near-tip block already known, the counter is unchanged.

### Inv routing

- **High-quality** (counter < threshold): inv messages get direct `GETDATA`.
- **Low-quality** (counter >= threshold): inv messages go through `GETHEADERS` first (inv → headers → getdata), adding a round-trip but avoiding blind block downloads.

This replaces Bitcoin's `sendheaders` optimisation, which Pearl does not use.

---

## Orphan Handling

Headers or blocks whose parent is unknown are silently ignored, as if never sent. If basic checks fail before searching for the parent, the peer may be punished. Otherwise, messages with an unknown `PrevBlock` are discarded — the same as receiving a block already known.

---

## Failure Handling

### Punishable (disconnect or ban)

- Non-contiguous headers in a HEADERS response.
- Inconsistent certificate inclusion within a single HEADERS message (ban).
- Invalid certificate (present but cryptographically wrong).
- Invalid difficulty transition.
- Timestamp monotonicity violation.
- Zero `ProofCommitment` in a header.

### Non-punishable (abort presync, mark peer low-quality)

- Missing certificates in a response to an `IncludeCertificates=true` request (PRESYNC threshold batches, spot-checks).
- Spot-check hash mismatch (peer may have answered a different query).
- Commitment mismatch during REDOWNLOAD.
- Incomplete message before reaching minimum work.

### REDOWNLOAD-specific punishable violations

- **Arriving REDOWNLOAD block's header bytes disagree with the Tier-1 approved header.** The peer approved one chain and is now trying to deliver a different one. Disconnect.
- **ZK certificate invalid at REDOWNLOAD block arrival.** `zkpow.VerifyCertificate` fails inside `chain.ProcessBlock`. Disconnect.
- **Non-REDOWNLOAD block from a peer during an active REDOWNLOAD session** (existing unrequested-block policy).

Peers that fail presync are marked low-quality by setting their counter above the threshold. Peers that do not help the node make progress may later be disconnected by ordinary stall/eviction logic, which is separate from presync.

---

## Locator Construction

Presync-driven `GETHEADERS` requests (PRESYNC, PRESYNC_SPOT_CHECK, REDOWNLOAD) use a locator built from two parts:

1. A **phase-specific tip hash**:
   - PRESYNC: `lastHeaderHash` (the presync tip).
   - PRESYNC_SPOT_CHECK: `spotCheckHeader.PrevBlock` (so the selected header is the first returned).
   - REDOWNLOAD: the REDOWNLOAD cursor hash (`redownloadCursor.hash` — stable across Tier-1 drain; see the *REDOWNLOAD tip cursor* section).
2. **Exponentially-spaced ancestors** of `chain_start`, matching Bitcoin Core's `LocatorEntries` format.

Event-driven `GETHEADERS` requests from outside the presync state machine use simpler single-hash locators:

- **Acceptance-tail probe** (`acceptValidatedHeaders`): `[lastAcceptedHash]` when the batch produced at least one new header, or `[headers[len-1].BlockHash()]` when every header in the batch was already in the block index.
- **Missing-cert retry** (sufficient-work batch with no certificates): `[firstHeader.PrevBlock]`, re-requesting the same range with `IncludeCertificates=true`.
- **Disconnected-batch backfill** (first header does not connect to anything we know): `LatestBlockLocator()`, the standard exponentially-spaced locator from our best header.
- **Initial sync** (`startSync`): `LatestBlockLocator()`.

### Rate-limiting

`GETHEADERS` messages to the same peer are rate-limited to one per `headersResponseTime` (2 minutes). The timestamp is tracked per peer as `lastGetHeadersTime`.

Every HEADERS message that is a plausible response to our outstanding `getheaders` clears the rate-limit timestamp on receipt, matching Bitcoin Core's `m_last_getheaders_timestamp` handling in `ProcessHeadersMessage` (see `net_processing.cpp` lines 2689, 2978, 3042). Three clear points cover all plausible-response cases:

1. **Empty HEADERS** (`handleHeadersMsg`): the peer is telling us they have nothing more; any follow-up request should be allowed immediately.
2. **Successful presync continuation** (`isContinuationOfLowWorkHeadersSync`): when `HeadersSyncResult.Success == true` the state machine has accepted the headers as extending its requested range, and the clear happens *before* the in-function `maybeSendGetHeaders`. A failed continuation (`!Success`, missing expected certs, etc.) deliberately does not clear — a peer sending garbage during presync cannot keep our rate limit perpetually open.
3. **Connecting non-presync HEADERS** (`handleHeadersMsg`): after the first header's fork point has been found in the block index (`chainStart != nil`), before the follow-up missing-cert retry or accept-tail probe fires. Disconnected batches (`chainStart == nil`) are not plausible responses to our outstanding `getheaders` and deliberately do not clear — a peer flooding us with plausible-looking but disconnected headers cannot amplify our outgoing getheaders.

The block-triggered presync-init path (`handleBlockMsg`: a low-work block starts a new presync session and immediately sends the first `GETHEADERS`) also clears the timestamp before the send, so the session's first request is never delayed by a stale prior timestamp.

Clearing the timestamp on receipt (rather than at each individual send site) means the follow-up `GETHEADERS` always flows through the generic `maybeSendGetHeaders` rate-limit gate — which still guards unprompted requests — without stalling event-driven continuations behind the 2-minute interval.

### Pipelined `GETHEADERS`

When an incoming PRESYNC or REDOWNLOAD batch extends the current phase tip, the node dispatches the **next** `GETHEADERS` *before* running the expensive per-header work on the current batch. One speculative request is in flight at a time. The peer starts producing the next batch while the local ZK verification of the current batch is still running; by the time verification completes, the next response is already on the wire or in the receive queue. This hides most of the local CPU cost behind the wire delivery of the next batch.

**Speculative dispatch preconditions.** At the entry of `handleHeadersMsg`, the node dispatches a speculative `GETHEADERS` only when **all** of the following hold:

- the peer has an active presync session (`state.headersSyncState != nil`);
- the session phase is PRESYNC or REDOWNLOAD (no speculation during PRESYNC spot-check or FINAL);
- the wire batch is full (`numHeaders == MAX_HEADERS_RESULTS`);
- the batch structurally continues the current phase tip (`headers[0].PrevBlock` equals `hss.lastHeaderHash` in PRESYNC, or the REDOWNLOAD cursor hash in REDOWNLOAD);
- no speculative request is already in flight (`state.pipelinedLocatorHead == nil`).

When dispatched, the locator is the phase-specific locator with `specTip = headers[len-1].BlockHash()` as the tip entry. `state.pipelinedLocatorHead` is set to `specTip`, and the `RequestMore`-driven `maybeSendGetHeaders` inside `isContinuationOfLowWorkHeadersSync` is suppressed for this batch.

**Stale-drop guard.** When a subsequent HEADERS message arrives and its `headers[0].PrevBlock == state.pipelinedLocatorHead`, the handler first computes the **currently expected prev-hash** from live post-processing state:

- no session → expected is zero;
- PRESYNC session → expected is `hss.lastHeaderHash`;
- REDOWNLOAD session → expected is the REDOWNLOAD cursor hash (exposed as `hss.RedownloadTipHash()`);
- PRESYNC spot-check or FINAL phase → expected is zero (these phases do not request follow-up headers non-speculatively).

If the expected prev-hash does not equal `state.pipelinedLocatorHead`, the speculative response is **silently dropped** (no processing, no disconnect, no peer-quality penalty). This exactly covers every case in which the non-speculative flow would have stopped before issuing the follow-up `GETHEADERS`: the previous batch's `ProcessNextHeaders` returned `!Success` (session aborted non-punishably), transitioned to FINAL without `RequestMore`, or failed a continuity / commitment check. If the expected prev-hash matches, `state.pipelinedLocatorHead` is cleared (consumed) and the batch flows through normal processing.

**DoS invariant.** *Speculation never causes the node to accept, validate beyond silent drop, or otherwise progress on any batch that the non-speculative flow would not have requested.* The stale-drop guard is the single enforcement point for this invariant. All other DoS mechanisms — anti-DoS work threshold, `maxCommitments` bound, cert threshold, spot-check selection, commitment cross-check during REDOWNLOAD, release-time certificate verification — run unchanged inside `ProcessNextHeaders` on every batch that is not dropped.

**Lifetime of `pipelinedLocatorHead`.** The field lives on `peerSyncState`, not inside `HeadersSyncState`, because it tracks a network-level expectation independent of the state machine's replay semantics. It is cleared when:

- the stale-drop guard consumes the matching speculative response (success path);
- the stale-drop guard drops a mismatching speculative response (DoS enforcement path);
- any non-speculative `GETHEADERS` is dispatched to the peer (initial sync, checkpoint-driven headers-first, inv-driven header probe, missing-cert retry, accept-tail probe, empty-batch retry, low-work init);
- the peer disconnects (`handleDonePeerMsg`);
- the presync session finalises.

Every direct `PushGetHeadersMsg` call in the node routes through `pushGetHeadersDirect`, which clears `state.pipelinedLocatorHead` and refreshes `state.lastGetHeadersTime` before sending. The rate-limited `maybeSendGetHeaders` similarly clears the field on any non-speculative successful send. This guarantees that a checkpoint-mode or inv-driven `GETHEADERS` cannot be confused with a pending speculative expectation, and correspondingly that the pipelining layer does not break correctness when mainnet `Checkpoints` are populated.

**Measured throughput (mainnet, wallet-node1, h=100 → h=2100, 100 headers per batch).**

> **Note.** The numbers below were captured against the previous REDOWNLOAD protocol, in which the peer was streaming certificates inside both HEADERS and BLOCK payloads. With cert-less REDOWNLOAD HEADERS and block-gated ZK verification (the current design), REDOWNLOAD throughput is expected to improve because the peer's uplink now carries each cert exactly once, and local CPU is no longer spent verifying certs twice. A fresh benchmark is pending (`bench` todo) and will replace this table.

| Phase     | Baseline      | Pipelined     | Δ      | Per-batch cycle      | Bottleneck                                               |
|-----------|---------------|---------------|--------|----------------------|----------------------------------------------------------|
| PRESYNC   | ~70 h/s       | ~103 h/s      | +47%   | cycle ≈ proc ≈ 735 ms | local header/commitment verification (CPU-bound)         |
| REDOWNLOAD| ~48 h/s       | ~45 h/s       | ~0%    | cycle ≈ 2.30 s       | peer-side upload bandwidth (block bodies served via `getdata` interleaved with REDOWNLOAD HEADERS replies *carrying duplicate certs*) |
| Full sync | ~78.7 s       | ~76.4 s       | -3%    | n/a                  | dominated by REDOWNLOAD's peer-side serving cost         |

The PRESYNC gain is close to the theoretical ceiling: once pipelining hides RTT, the cycle is proc-bound (verify + commitment write per 100 headers). REDOWNLOAD in the previous design saturated the peer's upload link streaming block bodies concurrently with cert-carrying HEADERS; with cert-less HEADERS the expected bottleneck shifts to local ZK verification at block-arrival time.

---

## State Machine

```
PRESYNC
  |
  +-- (spot-check selected, cert not in batch) --> PRESYNC_SPOT_CHECK --> PRESYNC
  |
  +-- (provisional_work >= minimum_required_work)
  v
REDOWNLOAD --> FINAL
```

`FINAL` is a terminal sentinel for the PRESYNC side of the state machine. A session reaches it when PRESYNC aborts or when REDOWNLOAD is dismissed by the `SyncManager` after the peer has signalled end-of-headers (`redownloadShortBatchSeen`) *and* both Tier-1 and Tier-2 have drained. Under the cert-less REDOWNLOAD design, REDOWNLOAD does not itself transition to FINAL on the header path; instead the `SyncManager` sets `state.headersSyncState = nil` when block-gated acceptance finishes. Subsequent headers from this peer re-enter the unified entry point at full strength — either taking the above-threshold path directly or starting a new presync on a different low-work fork.

Because REDOWNLOAD accepts its cert-verified blocks through the dedicated block-arrival path (`handleRedownloadBlockArrival`), the ordinary acceptance tail is only invoked for above-threshold batches. The acceptance tail (`acceptValidatedHeaders`) composes a single `shouldProbe` signal and, when it is set, issues a follow-up `GETHEADERS`. `shouldProbe` is true when any of the following holds on the current HEADERS batch:

- The wire batch was at `MAX_HEADERS_RESULTS` (a full batch, the usual progression case).
- The presync state machine transitioned to `FINAL` on this batch (even if the wire batch was shorter than `MAX_HEADERS_RESULTS`).

The probe only fires when all of the following are also true:

- No presync session is active (`state.headersSyncState == nil`) — while presync is active, the state machine drives its own `GETHEADERS` via `RequestMore`.
- `peer.LastBlock() > peerHeight`, i.e. the peer advertises a tip strictly above the height we are about to use as the probe's locator head. This suppresses the probe when the peer is already at or behind us.

The probe's locator head is chosen as follows:

- At least one header in the batch was new: `lastAccepted.Hash` (the last header newly inserted into the block index) and `peerHeight = lastAccepted.Height`.
- The batch contained no new headers (all already in the block index): the last header of the wire batch (`headers[len-1].BlockHash()`) and `peerHeight = peer.LastBlock()`. This matches Bitcoin Core's `GetLocator(pindexLast)` probe in `ProcessHeadersMessage` and is cheap because `AcceptBlockHeader` short-circuits on already-known headers without re-running certificate verification.

This unified tail lets the node transparently advance past the presync tip without any additional prodding, and keeps overlapping-batch progression working when a peer re-hands us headers we already accepted from someone else.

---

## Constants

| Name | Value | Description |
| --- | --- | --- |
| `MAX_HEADERS_RESULTS` | 100 | Headers per HEADERS message |
| `SPOT_CHECK_INV_PROB` | 1000 | Per-batch spot-check range |
| `redownloadApprovedCap` | 500 | Tier-1 REDOWNLOAD approved-headers FIFO cap (`HeadersSyncState`) |
| `redownloadApprovedHeadroom` | 200 | Free slots required in Tier-1 before `RequestMore=true` |
| `redownloadPendingCap` | 100 | Tier-2 REDOWNLOAD block-fetch buffer cap (`peerSyncState`) |
| `MIN_BLOCK_INTERVAL` | 1 s | Strict timestamp monotonicity |
| `TIMESTAMP_WINDOW` | `MaxTimeOffsetMinutes * 60` | Max future offset (300 s on mainnet) |
| `antiDoSBufferBlocks` | 100 | Work deduction for anti-DoS threshold |
| `peerQualityThreshold` | 5 | Non-near-tip announcements before low-quality |
| `headersResponseTime` | 2 min | Min interval between getheaders to same peer |

---

## Differences from Bitcoin Core

| Aspect | Bitcoin Core | Pearl |
| --- | --- | --- |
| PoW location | In header | In separate certificate |
| PoW check cost | Trivial | Expensive, requires gating |
| Proof object size | None beyond header | ~60 KB |
| `GETHEADERS` | `locator + hashStop` | `locator + hashStop + IncludeCertificates` |
| PRESYNC per-header checks | Continuity, permitted difficulty, sparse commitments, cheap PoW | Continuity, timestamp monotonicity, exact WTEMA difficulty, ProofCommitment well-formedness, dense 1-bit commitments, provisional work |
| PRESYNC PoW verification | Cheap for every header | Above-threshold batches + random spot-checks |
| PRESYNC work meaning | Effectively validated | Provisional |
| Commitment density | Sparse | 1 bit per header |
| Spot-check state | None | `PRESYNC_SPOT_CHECK` |
| REDOWNLOAD cert requests | N/A | `IncludeCertificates=false`; certificate travels inside the block body |
| REDOWNLOAD PoW verification | Not needed | Required before permanent acceptance |
| Permanent header acceptance | After redownload consistency | After redownload consistency + valid certificate |
| Block contents | Header + txs | Header + certificate + txs |
| Tip announcement | `sendheaders` | Quality-based inv routing |
