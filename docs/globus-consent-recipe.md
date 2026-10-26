# Globus SDK transfer recipe (dawn ↔ dusk GCP) — for a human-driven session

Status 2026-10-26: rsync-over-ssh is the working path (alex directive);
Globus PROPER is a follow-up needing one browser consent click. Recipe below
is what ops_sh worked out — reuse it when someone can spare the click.

## Facts (verified live)

- OAuth **client id** (native app registered 2026-10-13, client-key in
  `.globusonline/lta/client-key.pem`): `2a1e2134-5125-495e-b36c-da093c1ca11e`
- **Endpoint/collection UUIDs ≠ client id** (gotcha that cost us a round):
  - dawn: `1d787e95-a611-4b49-9457-0ff53b1c8b14` (from register.log /
    systemd unit description; `lta/client-id.txt` is the CLIENT uuid)
  - dusk (rail_dusk): `68883a62-dbe7-483e-adab-f3c1046d43ba`
- Both GCP v3.3.1, relay relay.globusonline.org:2223, active.
  dawn managed by systemd user unit `globus-connect-personal.service`
  (restart with `systemctl --user restart globus-connect-personal.service`;
  `gc-ctrl.py` has no `restart` subcommand).
- Allowed paths: dawn `/mnt/d/globus,1,1` + `/mnt/d/ops/sh/data,0,1` (added
  2026-10-26, config-paths.bak-sh is the backup); dusk `/home/dusk/d/globus,1,1`
  + `/home/dusk/Downloads,1,1`.
- The GCP client-key JWT-bearer grant is NOT accepted by
  auth.globus.org `/v2/oauth2/token` (400 unsupported_grant_type for every
  kid/aud variant) — GCP keys are data-plane only. No Transfer-API tokens
  exist anywhere on the fleet (dusk `~/.globus/cli/storage.db` token table
  is empty).

## Consent flow (the one click)

1. Ready-made flow script: `ops_sh` scratch
   `globus_oauth_flow.py` (venv `/mnt/d/ops/sh/.venv-globus`, globus-sdk 4.10 +
   pyjwt + cryptography). It prints an AUTH_URL, serves the loopback redirect
   on 127.0.0.1:4916, exchanges the code, saves tokens to
   `/home/dawn/.globusonline/lta/transfer_token.json`.
2. A human opens the AUTH_URL in ANY browser that (a) has a logged-in
   Globus session and (b) can reach the machine running the callback server
   (redirect target 127.0.0.1 — so same machine, or run the script there).
   One click on Allow.
3. Scopes requested (already in the script): transfer.all + offline_access +
   openid/profile/email + data_access for BOTH collection UUIDs above.
   Without per-collection data_access, transfers fail CONSENT_REQUIRED
   (see env_exp skill explorer-dispatch/references/globus.md).
4. Then submit with TransferClient: NativeAppAuthClient → refresh token →
   TransferData(submit_transfer(source_endpoint, dest_endpoint, ...)),
   sync_level=checksum, notify=false. Poll task events for checkpoints.
