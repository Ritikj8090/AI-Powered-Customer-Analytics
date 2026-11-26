#!/usr/bin/env bash
# Run every phase in order. Usage:
#   ./run_pipeline.sh            # everything (sentiment on a 5,000-review sample)
#   ./run_pipeline.sh --full     # sentiment on all reviews (slow on a laptop)
set -e
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
SENTIMENT_ARGS="--limit 5000"
[[ "$1" == "--full" ]] && SENTIMENT_ARGS=""

echo "== Phase 1: load data + KPIs ==";          $PY src/load_data.py && $PY src/run_kpis.py > /dev/null
echo "== Phase 2: segmentation ==";              $PY src/segmentation.py
echo "== Phase 3: churn + review models ==";     $PY src/churn_model.py && $PY src/churn_model.py --target negative_review
echo "== Phase 4: forecasting ==";               $PY src/forecasting.py
echo "== Phase 5: review sentiment ==";          $PY src/sentiment.py $SENTIMENT_ARGS

if grep -qE '^LLM_PROVIDER=' .env 2>/dev/null; then
  echo "== Phase 6: text-to-SQL evaluation =="; $PY src/text_to_sql.py --eval
else
  echo "== Phase 6 skipped: set LLM_PROVIDER in .env, then run: $PY src/text_to_sql.py --eval"
fi

echo
echo "Done. Start the dashboard with:  streamlit run app/streamlit_app.py"
