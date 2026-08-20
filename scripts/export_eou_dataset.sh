#!/usr/bin/env bash
#
# Export the EOU dataset from the backend Postgres to JSONL, and (optionally)
# download the matching call recordings from Azure Blob.
#
# Run this AFTER the prod deployment, once calls have accumulated on the
# EOU_DATASET_CAPTURE-enabled test org(s). The EOUDetector repo has NO psql
# access. This script produces flat files that the Python tooling then ingests.
#
# One JSONL line per call: the EOU capture (slim meta + nested turns) joined to
# receptionist_calls for the normalized meta we deliberately did NOT duplicate
# (philosophy A: single source of truth = receptionist_calls, joined by uuid).
#
# ── PHI WARNING ─────────────────────────────────────────────────────────────
# The output contains patient speech (caller_text, asr_partials) and phone
# numbers. It is git-ignored (data/, *.jsonl, *.wav). Keep it local. Do not push.
#
# ── Config (env) ────────────────────────────────────────────────────────────
#   EOU_DATABASE_URL                 psql connection string to the backend DB
#   AZURE_STORAGE_CONNECTION_STRING  (only for --with-audio) from CareCallHouseMade/.env
#   CONTAINER_NAME                   (only for --with-audio) from CareCallHouseMade/.env
#
# ── Usage ───────────────────────────────────────────────────────────────────
#   ./scripts/export_eou_dataset.sh [--since YYYY-MM-DD] [--org <uuid>]
#                                   [--limit N] [--out DIR] [--with-audio]
#
# Examples:
#   EOU_DATABASE_URL=postgres://... ./scripts/export_eou_dataset.sh
#   ./scripts/export_eou_dataset.sh --since 2026-07-23 --with-audio
#
set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
SINCE=""
ORG=""
LIMIT=""
OUT=""
WITH_AUDIO=0

# ── args ────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --since)      SINCE="$2"; shift 2 ;;
    --org)        ORG="$2"; shift 2 ;;
    --limit)      LIMIT="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --with-audio) WITH_AUDIO=1; shift ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

: "${EOU_DATABASE_URL:?set EOU_DATABASE_URL to the backend Postgres connection string}"
command -v psql >/dev/null || { echo "psql not found" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-data/eou_export_${STAMP}}"
mkdir -p "$OUT"
CALLS_FILE="$OUT/eou_calls.jsonl"

# ── build the WHERE clause from filters ─────────────────────────────────────
WHERE="WHERE TRUE"
[[ -n "$SINCE" ]] && WHERE="$WHERE AND c.created_at >= '${SINCE}'::timestamptz"
[[ -n "$ORG"   ]] && WHERE="$WHERE AND c.organization_id = '${ORG}'::uuid"
LIMIT_SQL=""
[[ -n "$LIMIT" ]] && LIMIT_SQL="LIMIT ${LIMIT}"

# ── the export query: one jsonb object per call, turns nested & ordered ─────
# eou_call_dataset (slim, EOU-specific) + eou_turn[] + receptionist_calls meta.
read -r -d '' QUERY <<SQL || true
SELECT jsonb_build_object(
  'uuid',                    c.uuid,
  'organization_id',         c.organization_id,
  'created_at',              c.created_at,
  'segmentation_timeout_ms', c.segmentation_timeout_ms,
  'ai_interrupted',          c.ai_interrupted,
  'human_interrupter',       c.human_interrupter,
  'audio_ref',               c.audio_ref,
  -- normalized meta joined from receptionist_calls (not duplicated on eou_call_dataset)
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
${WHERE}
ORDER BY c.created_at
${LIMIT_SQL};
SQL

echo "→ exporting EOU calls to $CALLS_FILE ..."
# -t tuples-only, -A unaligned  => one JSON object per line = JSONL
psql "$EOU_DATABASE_URL" -t -A -o "$CALLS_FILE" -c "$QUERY"
# strip any blank trailing line
sed -i.bak '/^[[:space:]]*$/d' "$CALLS_FILE" && rm -f "$CALLS_FILE.bak"

N_CALLS=$(wc -l < "$CALLS_FILE" | tr -d ' ')
echo "✓ $N_CALLS calls exported → $CALLS_FILE"

# ── optional: download the caller recordings from Azure Blob ────────────────
if [[ "$WITH_AUDIO" -eq 1 ]]; then
  : "${AZURE_STORAGE_CONNECTION_STRING:?set AZURE_STORAGE_CONNECTION_STRING (CareCallHouseMade/.env) for --with-audio}"
  : "${CONTAINER_NAME:?set CONTAINER_NAME (CareCallHouseMade/.env) for --with-audio}"
  command -v az >/dev/null || { echo "az CLI not found" >&2; exit 1; }
  command -v jq >/dev/null || { echo "jq not found (needed to read audio_ref)" >&2; exit 1; }

  AUDIO_DIR="$OUT/audio"
  mkdir -p "$AUDIO_DIR"
  echo "→ downloading recordings to $AUDIO_DIR ..."
  DL=0; SKIP=0
  while IFS= read -r ref; do
    [[ -z "$ref" || "$ref" == "null" ]] && continue
    dest="$AUDIO_DIR/$ref"
    if [[ -f "$dest" ]]; then SKIP=$((SKIP+1)); continue; fi
    if az storage blob download \
        --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
        --container-name "$CONTAINER_NAME" \
        --name "$ref" --file "$dest" --no-progress >/dev/null 2>&1; then
      DL=$((DL+1))
    else
      echo "  ! missing blob: $ref" >&2
    fi
  done < <(jq -r '.audio_ref // empty' "$CALLS_FILE")
  echo "✓ audio: $DL downloaded, $SKIP already present → $AUDIO_DIR"
fi

echo
echo "Done. Ingest with the EOUDetector tooling, e.g.:"
echo "  python -m eou_detector.dataset build --export $CALLS_FILE --audio-dir $OUT/audio"
