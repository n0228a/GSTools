# GSTools — Claude Code Guidelines

## Git workflow

Always stage and commit the current working state before applying any batch of changes (refactors, simplify passes, feature additions). This ensures every change set can be reverted independently with `git checkout` or `git revert` if it introduces a regression.
