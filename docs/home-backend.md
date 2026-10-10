# Local home backend

The future app connects to the APM API. It registers a Homebridge or Matter Server integration, lists the devices already known to that controller, and saves selected devices with a friendly name, room, and aliases. Gemma receives those names and supported actions, then calls tools using stable device IDs. Python validates the complete tool batch before dispatching any device request.

This first version is a local backend foundation. It does not install a controller, commission Matter devices, manage Thread credentials, scan the LAN, provide an app interface, or offer a chat endpoint. Matter pairing belongs to the existing controller for now. Discovery only proposes devices; it does not register or control them.

## Try the API with simulated devices

From the project directory:

```sh
uv pip install --python .venv/bin/python -e '.[home]'
.venv/bin/apm-server --registry work/home.json --init
```

The `--init` flag creates a registry containing a simulated kitchen light and garage. It fails if the file already exists. On subsequent starts:

```sh
.venv/bin/apm-server --registry work/home.json
```

Open [API documentation](http://127.0.0.1:8765/docs), or use another terminal:

```sh
curl http://127.0.0.1:8765/v1/devices
curl -X POST http://127.0.0.1:8765/v1/devices/kitchen_lights/commands \
  -H 'Content-Type: application/json' -d '{"state":"off"}'
curl http://127.0.0.1:8765/v1/devices/kitchen_lights/state
curl http://127.0.0.1:8765/v1/context
```

Use that registry with the existing assistant:

```sh
.venv/bin/apm --home-config work/home.json --check-home
.venv/bin/apm --home-config work/home.json --text 'Turn off the kitchen lights'
```

For voice, add `--home-config work/home.json` to your existing command, retaining your personal `--wake-model` and `--keep-alive=-1m` settings. Omitting `--home-config` preserves the original simulation. `--demo` always uses that original simulation and rejects a home configuration.

Simulation state is in memory and separate in each process; the API and CLI share registry metadata, not simulated state. Physical state is read from the configured controller on each operation. Run one API process as the registry writer, without multiple workers. JSON updates use atomic replacement with owner-only permissions; other processes reload the registry before each request. Registry changes during model inference reject the stale tool batch so a renamed or rebound device cannot silently receive an old command.

## Configure an existing controller

The example [home.example.json](../examples/home.example.json) contains both integration types and no registered devices. It validates without installed credentials and makes no connection until discovery, a read, or a command is requested. Use actual server addresses for your home.

Homebridge connects to the **Homebridge UI HTTP API**, not the HAP accessory port. [Homebridge accessory control](https://github.com/homebridge/homebridge-config-ui-x/wiki/Enabling-Accessory-Control) must already be enabled by its owner. This adapter never changes that setting. Use either `token_env` or both `username_env` and `password_env` to name environment variables available to the API/voice process. Raw credentials and URL-embedded credentials are rejected in configuration. Environment files are not loaded automatically. Password login with interactive two-factor authentication is unsupported; use a valid UI bearer token. Supplied tokens need replacement after they expire.

Example integration request body for `PUT /v1/integrations/homebridge`:

```json
{
  "type": "homebridge",
  "url": "http://127.0.0.1:8581",
  "token_env": "APM_HOMEBRIDGE_TOKEN",
  "timeout": 5
}
```

Set the referenced environment variable before starting each process that connects to Homebridge. Then call `GET /v1/integrations/homebridge/discover`. Copy a returned `target.unique_id`; this is a Homebridge UI service identifier, not a HomeKit pairing code. `available` describes readable/writable characteristics, not guaranteed physical reachability. An empty list can also reflect a controller discovery problem.

Matter uses the [Matter Server WebSocket API](https://github.com/matter-js/python-matter-server/blob/main/docs/websockets_api.md), schema 11 compatible, against an existing controller with already commissioned devices. A Matter-over-Thread setup also needs an appropriate Thread network/border router. A device paired into another ecosystem is not automatically known to this controller. Do not put the unauthenticated controller WebSocket on a public network.

Example body for `PUT /v1/integrations/matter`:

```json
{"type":"matter","url":"ws://127.0.0.1:5580/ws","timeout":10}
```

Call `GET /v1/integrations/matter/discover`. It lists the controller's commissioned On/Off endpoints and returns `target.node_id` and `target.endpoint`. Initial support is On/Off only: no brightness, thermostats, locks, covers, or Matter garage control. Homebridge Switch/Outlet and Matter On/Off endpoints use the initial `light` kind; give them accurate friendly names, such as “Desk plug.”

## Register selected devices and context

After discovery, use `PUT /v1/devices/desk_lamp` with a body like this, replacing the target with the actual discovery result:

```json
{
  "name": "Desk lamp",
  "kind": "light",
  "room": "Office",
  "aliases": ["work light", "office lamp"],
  "integration": "matter",
  "target": {"node_id": 123, "endpoint": 1}
}
```

For Homebridge, set `integration` to the registered Homebridge integration ID and use its returned `{"unique_id":"..."}` target instead. A garage service uses `kind: "garage"`. IDs use lowercase letters, numbers, and underscores, starting with a letter, up to 64 characters. This initial registry supports up to 16 integrations and 64 devices. PUT replaces the full configuration; keep fields you wish to retain.

Only registered devices appear in the model's tool enums. Its catalog contains ID, name, room, aliases, kind, supported states, and whether the device is simulated. Server URLs, credential references/values, and protocol targets are omitted. Catalog strings are marked as untrusted descriptive data. Current state is queried through a tool rather than assumed from old conversation history. The model is instructed to clarify ambiguous names or rooms.

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | API process health; does not check controllers |
| GET / PUT | `/v1/integrations` / `/v1/integrations/{id}` | List integrations or register/replace one |
| GET | `/v1/integrations/{id}/discover` | List controller devices without registration |
| GET | `/v1/devices` | Registered metadata and bindings |
| PUT / DELETE | `/v1/devices/{id}` | Register/replace or remove a device |
| GET | `/v1/devices/{id}/state` | Read current state |
| POST | `/v1/devices/{id}/commands` | Explicit `{"state":"on"}` / `off` / `open` / `closed` |
| GET | `/v1/context` | Model-visible catalog, tools, and registry revision |

The server binds only to `127.0.0.1`. It rejects nonlocal Host headers. Browser requests must match the serving origin's scheme, hostname, and effective port; a different localhost port or loopback hostname is rejected, even with a valid bearer token. CLI requests without an Origin header remain supported. Optional `APM_API_TOKEN` bearer authentication protects device routes and health. Public API schemas/docs contain no registry values; use the same-origin Swagger page's Authorize button when a token is configured. In the initial local development setup, omitting the token allows local callers. Remote mobile access, user accounts, a credential vault, and durable audit history still need implementation before deployment.

Command results include `accepted`, `requested_state`, `state`, `simulated`, and `ok`. `ok` means the adapter operation returned a result, not that the requested final physical state was achieved. A garage can be `opening`/`closing`; an accepted command can have `state: "unknown"` if readback failed. Timed-out writes are never automatically retried because the physical action may already have happened. After an execution error or unconfirmed write, remaining LLM tool calls are skipped. Batches are prevalidated but cannot be rolled back across hardware. An API 502 likewise leaves the physical outcome uncertain; read the device before retrying.

## Verification

```sh
uv pip install --python .venv/bin/python httpx
.venv/bin/python -m unittest discover -s tests
```

Tests use fake Homebridge HTTP and Matter WebSocket transports, plus an in-process API client. They cover routing, invalid batches, credential omission, explicit registration, persistence, and acknowledged versus observed state. They do not commission or operate physical devices.
