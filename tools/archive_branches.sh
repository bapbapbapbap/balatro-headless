#!/usr/bin/env bash
# Archives the repo's history into tags (ARCHIVE.md), then optionally deletes every remote branch but main.
# Run from chud's clone (cloud sessions can't push tags or delete branches), after the 2026-11-25 cleanup is merged.
#
#   bash tools/archive_branches.sh            # creates and pushes the tags; lists what deleting would remove
#   bash tools/archive_branches.sh --delete   # the same, then deletes the branches whose heads are inside a tag
#
# A branch is deleted only if its head is reachable from one of the tags. Any branch whose head isn't already
# inside archive-2026-11-25 gets its own tag archive-branch-<name> first, so nothing becomes unreachable.
set -euo pipefail
cd "$(dirname "$0")/.."
DELETE=0; [ "${1:-}" = "--delete" ] && DELETE=1

git fetch --quiet --prune --tags origin

# archive-2026-11-25: main with every open branch merged, just before the cleanup removed past runs' files.
ARCH=ef3c0de675cdeac13ac2925e3b8f0ad944d14e0c
git rev-parse -q --verify "$ARCH^{commit}" >/dev/null || { echo "$ARCH not found: merge the cleanup PR first"; exit 1; }
git merge-base --is-ancestor 6ffd546 "$ARCH" && git merge-base --is-ancestor 53548d9 "$ARCH" \
  && git merge-base --is-ancestor 9ce56e2 "$ARCH" || { echo "$ARCH doesn't contain all three branches"; exit 1; }

declare -A TAGS=(
  [archive-2026-11-25]=$ARCH
  [run-v4c]=e16a63d [run-v5]=e1504ad [run-v6]=8a11d21 [run-v6c]=fee0e8b
  [run-v6e]=ed75c1d [run-v6d]=e0ff1f5 [run-v6g]=74eeab7
)
for t in "${!TAGS[@]}"; do
  if git rev-parse -q --verify "refs/tags/$t" >/dev/null; then echo "tag $t exists"; continue; fi
  git rev-parse -q --verify "${TAGS[$t]}^{commit}" >/dev/null || { echo "commit ${TAGS[$t]} for $t not found"; exit 1; }
  git tag "$t" "${TAGS[$t]}" && echo "tag $t -> $(git rev-parse --short ${TAGS[$t]})"
done

ALLTAGS=$(git tag)
for b in $(git for-each-ref --format='%(refname:strip=3)' refs/remotes/origin | grep -vx 'HEAD\|main'); do
  head=$(git rev-parse "origin/$b")
  inside=""
  for t in $ALLTAGS; do git merge-base --is-ancestor "$head" "$t" && { inside=$t; break; }; done
  if [ -z "$inside" ]; then
    t="archive-branch-$b"
    git tag "$t" "$head" && inside=$t && echo "tag $t -> $(git rev-parse --short $head) (head not in any tag)"
    ALLTAGS="$ALLTAGS $t"
  fi
  echo "branch $b ($(git rev-parse --short $head)) is kept by tag $inside"
done
git push --quiet origin --tags && echo "tags pushed"

if [ $DELETE = 1 ]; then
  for b in $(git for-each-ref --format='%(refname:strip=3)' refs/remotes/origin | grep -vx 'HEAD\|main'); do
    git push --quiet origin --delete "$b" && echo "deleted origin/$b"
  done
  git fetch --quiet --prune origin
else
  echo "dry run: nothing deleted (rerun with --delete)"
fi
