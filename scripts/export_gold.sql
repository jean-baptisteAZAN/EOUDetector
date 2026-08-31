-- ===========================================================================
-- export_gold.sql : export the gold EOU dataset from the backend Postgres.
--
-- Produces one JSON object per call (slim EOU meta + nested, ordered turns),
-- in the exact shape that eou_detector/eval/gold_to_eotbench.py ingests.
--
-- USAGE (JSONL to a file, one call per line):
--   psql "$EOU_DATABASE_URL" -t -A -f scripts/export_gold.sql > data/gold/eou_calls.jsonl
--     -t  : tuples only (no header/footer)
--     -A  : unaligned (no column padding)  => one JSON object per line = JSONL
--
-- BEFORE RUNNING: replace the organization id placeholder below.
--   :ORG  ->  'XXXXXXXXXXXXX'      (replace XXXXXXXXXXXXX with the real org uuid)
-- Optionally uncomment the date filter to restrict to recent calls.
--
-- PHI WARNING: the output contains patient speech (caller_text, asr_partials)
-- and phone numbers. Keep it local, git-ignored (data/, *.jsonl). Do not push.
--
-- FULL PIPELINE TO NEW METRICS (run after this export):
--   1. audio:   download the stereo recordings by audio_ref from Azure Blob
--               (see scripts/export_eou_dataset.sh --with-audio) into data/gold/audio/
--   2. convert: python -m eou_detector.eval.gold_to_eotbench \
--                 --export data/gold/eou_calls.jsonl \
--                 --audio-dir data/gold/audio \
--                 --out data/gold_eotbench_fr
--   3. score:   point the eot-bench harness --path at data/gold_eotbench_fr and
--               rerun predict + compute-metrics + compare-models (see docs/reproduce.md).
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- SANITY / VOLUME CHECKS (run these first, individually, before the export).
-- They are commented out so `-f` only runs the export below. Uncomment one at
-- a time (replace XXXXXXXXXXXXX) to check you have enough data and a sane
-- eot/hold balance before converting.
-- ---------------------------------------------------------------------------
-- -- how many calls and turns for the org:
-- SELECT count(DISTINCT c.uuid) AS calls, count(t.*) AS turns
-- FROM eou_call_dataset c
-- JOIN eou_turn t ON t.uuid = c.uuid
-- WHERE c.organization_id = 'XXXXXXXXXXXXX'::uuid;
--
-- -- eot vs hold balance (eot = caller did NOT resume; hold = caller resumed):
-- SELECT
--   count(*) FILTER (WHERE t.caller_resumed_after_ms IS NULL) AS eot,
--   count(*) FILTER (WHERE t.caller_resumed_after_ms IS NOT NULL) AS hold
-- FROM eou_call_dataset c
-- JOIN eou_turn t ON t.uuid = c.uuid
-- WHERE c.organization_id = 'XXXXXXXXXXXXX'::uuid;
--
-- -- calls that actually have audio (audio_ref set), needed for scoring:
-- SELECT count(*) AS calls_with_audio
-- FROM eou_call_dataset c
-- WHERE c.organization_id = 'XXXXXXXXXXXXX'::uuid AND c.audio_ref IS NOT NULL;

-- ---------------------------------------------------------------------------
-- THE EXPORT QUERY (this is what `-f` runs and pipes to JSONL).
-- One jsonb object per call; turns nested and ordered by turn_index.
-- ---------------------------------------------------------------------------
SELECT jsonb_build_object(
  'uuid',                    c.uuid,
  'organization_id',         c.organization_id,
  'created_at',              c.created_at,
  'segmentation_timeout_ms', c.segmentation_timeout_ms,
  'ai_interrupted',          c.ai_interrupted,
  'human_interrupter',       c.human_interrupter,
  'audio_ref',               c.audio_ref,
  -- normalized meta joined from receptionist_calls (single source of truth):
  'motif',                   rc.motif,
  'started_at',              rc."startedAt",
  'caller_number',           rc.number,
  'is_success',              rc."isSuccess",
  'is_transfered',           rc."isTransfered",
  'turns', COALESCE((
    SELECT jsonb_agg(jsonb_build_object(
      'turn_index',              t.turn_index,
      'caller_text',             t.caller_text,
      'asr_partials',            t.asr_partials,
      't_pause_start_ms',        t.t_pause_start_ms,
      't_decision_ms',           t.t_decision_ms,
      't_agent_reply_ms',        t.t_agent_reply_ms,
      'silence_ms',              t.silence_ms,
      'endpoint_mechanism',      t.endpoint_mechanism,
      'caller_resumed_after_ms', t.caller_resumed_after_ms,
      'was_interrupted',         t.was_interrupted,
      'hardcase_tag',            t.hardcase_tag,
      'features',                t.features
    ) ORDER BY t.turn_index)
    FROM eou_turn t WHERE t.uuid = c.uuid
  ), '[]'::jsonb)
)
FROM eou_call_dataset c
LEFT JOIN receptionist_calls rc ON rc.id_call = c.uuid
WHERE c.organization_id = '5b936e22-35a0-4606-932a-52749a8d4420'::uuid
  -- AND c.created_at >= '2026-08-01'::timestamptz   -- optional: recent calls only
ORDER BY c.created_at;
