# Local-first CI and durable evidence

Owner policy, 2026-10-27: `owner-local-ci-20261027-v1`.

1. Run the complete required unit, privacy, regression, build and browser checks
   locally. Keep useful partial outcomes every 20-60 seconds. Reuse existing
   source-bound local-QA tooling; this document does not create a success receipt.
2. Before online work inspect this repository's `ci_policy.json`, required checks,
   exact tree, event/branch and every matrix job. Routine online execution,
   including self-hosted Actions, is limited to 60 seconds. A long legacy timeout
   is not permission to run it. First adapt to an authentic short local-QA gate
   or obtain a specific owner exception; never skip tests or fake a green status.
3. Private jobs accept only repository-scoped, deliberately started, ephemeral
   self-hosted runners with the workflow label recorded in `ci_policy.json`.
   Keep OS/matrix distinctions. Do not use organization-wide broad runners or
   leave a persistent runner enabled for arbitrary/fork PR code. Never place an
   owner OAuth credential in Actions. If the host is offline, report QUEUED or
   NOT_RUN with the required label; there is no paid GitHub-hosted fallback.
4. Existing long experimental/private QA recipes are manual-only. Their commands,
   inputs, assertions, OS matrix, dependency pins and evidence retention remain
   intact. They are not declared passed, qualified, or licensed for a long online
   run. The native short publication/validation/control gates remain automatic.
5. Standard hosted runners in an already-public repository may be used while
   verified free; larger/GPU/custom-image runners are not automatically free.
   The local-first and one-minute rules still apply. Do not make private code
   public to avoid a bill. Check shared owner/organization usage before dispatch.
6. Do not increase a budget automatically. Server-side zero-compute budgets and
   owner alerts are intentional. On a budget/quota block use local execution and
   report the actual blocker. A product ceiling is not standing spend permission.
7. Actions artifacts are short-lived transport. Disposable QA/publication
   duplicates should normally live 1 day, up to 3 with a stated recovery need.
   Unique scientific outputs, recovery packages and failed/null evidence must
   first be classified, preserved in the appropriate private Release, freshly
   downloaded and checked by exact bytes/SHA256 and recovery before cleanup.
   Do not blindly shorten old retention or delete other work by age.
8. Use Git for source and private project Releases for large immutable evidence.
   Split Release files below 2 GiB. Git/LFS, Release assets, Actions artifacts,
   cache and local disks are different limits. Never publish recordings, private
   QA/source or credentials to the public website. Preserve existing privacy and
   publication boundaries and actual required-check names.

The owner's global guard and detailed account controls are maintained privately
in `Gradient19/Observations-draft/operations/github-cost-control/` and the owner's
global agent instructions. Do not copy account billing amounts into public docs.
This guidance is not a continuously running monitor or permission to send mail.
