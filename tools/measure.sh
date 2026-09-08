#!/bin/bash
# Standard validity measurement for one project, run inside its ffe-<pj>
# container: the same corpus rules as the typical fuzzing command
# (--bug-corpus seeds included) and the fixed per-sample seed list, so runs
# are comparable with the HEAD baseline rows in output/validrate/history.tsv.
#
#   tools/measure.sh <project> <tag> [extra validrate args...]
#
# Runs s1 (with failing-child dumps) then s2. Logs: output/logs/<pj>-<tag>-s{1,2}.log
set -u
pj=$1; tag=$2; shift 2
case $pj in swift|flang|tint) w=/work/fusion-fuzz;; *) w=/home/fuzz/WorkSpace/fusion-fuzz;; esac
if docker exec ffe-$pj pgrep -f "[v]alidrate.py --project $pj" >/dev/null; then
  echo "ffe-$pj: a validrate run is already active; not launching" >&2; exit 1
fi
docker exec -d ffe-$pj bash -c "cd $w && \
  python3 tools/validrate.py --project $pj --tag $tag --sample-seed 1 --bug-corpus --sample-file output/validrate/$pj-sample-s1.json --dump-fail 6 $* > output/logs/$pj-$tag-s1.log 2>&1 && \
  python3 tools/validrate.py --project $pj --tag $tag --sample-seed 2 --bug-corpus --sample-file output/validrate/$pj-sample-s2.json $* > output/logs/$pj-$tag-s2.log 2>&1"
echo "launched $pj $tag"
