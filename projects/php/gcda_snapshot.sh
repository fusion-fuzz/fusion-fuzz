#!/usr/bin/env bash
# Save or restore all .gcda files under php-src, preserving relative paths.
#
# Usage:
#   ./gcda_snapshot.sh save   [snapshot_dir]   # default: gcda_snapshot/
#   ./gcda_snapshot.sh restore [snapshot_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHP_SRC="$SCRIPT_DIR/php-src"
SNAPSHOT_DIR="${2:-$SCRIPT_DIR/gcda_snapshot}"

cmd="${1:-}"

case "$cmd" in
  save)
    echo "Saving .gcda files to $SNAPSHOT_DIR ..."
    rm -rf "$SNAPSHOT_DIR"
    find "$PHP_SRC" -name "*.gcda" | while read -r f; do
      rel="${f#$PHP_SRC/}"
      dest="$SNAPSHOT_DIR/$rel"
      mkdir -p "$(dirname "$dest")"
      cp "$f" "$dest"
    done
    count=$(find "$SNAPSHOT_DIR" -name "*.gcda" | wc -l)
    echo "Saved $count .gcda files."
    ;;

  restore)
    if [[ ! -d "$SNAPSHOT_DIR" ]]; then
      echo "Error: snapshot directory not found: $SNAPSHOT_DIR" >&2
      exit 1
    fi
    echo "Restoring .gcda files from $SNAPSHOT_DIR ..."
    find "$SNAPSHOT_DIR" -name "*.gcda" | while read -r f; do
      rel="${f#$SNAPSHOT_DIR/}"
      dest="$PHP_SRC/$rel"
      mkdir -p "$(dirname "$dest")"
      cp "$f" "$dest"
    done
    count=$(find "$SNAPSHOT_DIR" -name "*.gcda" | wc -l)
    echo "Restored $count .gcda files."
    ;;

  *)
    echo "Usage: $0 {save|restore} [snapshot_dir]"
    exit 1
    ;;
esac
