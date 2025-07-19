# 02\_mcp\_integration.md

# MCP STDIO Integration Guide

This guide explains how to register and integrate the **Avatar Renderer Pod** as an MCP tool in your MCP Gateway, enabling programmatic, low-latency tool access via the STDIO transport protocol.

---

## 1 · Prerequisites

* A running MCP Gateway instance (e.g. [CluedIn MCP Gateway](https://github.com/mcpgateway/mcpgateway))
* Administrator credentials or JWT for the MCP Gateway Admin API
* Docker image of `avatar-renderer-pod` built and pushed, or a local container instance
* `mcp-tool.json` manifest available under the `avatar-renderer-pod/` directory

---

## 2 · Enable MCP mode in the pod

Ensure the container starts the MCP server entrypoint instead of (or alongside) the REST API. In your Kubernetes Helm values or raw deployment:

```yaml
containers:
  - name: avatar-renderer
    image: <REGISTRY>/avatar-renderer:<TAG>
    args: ["/app/mcp_server.py"]    # Launch MCP STDIO server
    env:
      - name: MCP_ENABLE
        value: "true"
      # REST API remains available if needed via api.py
```

---

## 3 · Registering via Admin UI

1. **Obtain Admin JWT**:

   ```bash
   export ADMIN_TOKEN=$(
     mcpctl token issue --username <admin> --secret <secret> --exp 3600
   )
   ```
2. **Open MCP Gateway Admin UI** in browser (e.g. `http://localhost:4444/admin`).
3. Navigate to **Catalog → Servers → Add Server**.
4. Fill in:

   * **Name**: `avatar_renderer`
   * **Transport**: `stdio`
   * **Command**: `/app/.venv/bin/python` (path inside container)
   * **Args**: `[/app/mcp_server.py]`
   * **Auto-discover**: ✓ (enabled)
5. Click **Save**. The gateway will launch the container, auto-detect the `render_avatar` tool, and list it under **Tools**.

---

## 4 · Registering via Admin API

Use the Admin API to automate registration:

```bash
curl -X POST http://<GATEWAY_HOST>/api/servers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d @mcp-tool.json
```

This will:

* Pull the `avatar-renderer-pod` image if not already present
* Start a STDIO server process
* Auto-discover and register the `render_avatar` tool

---

## 5 · Invoking the Tool

Once registered, clients can invoke the tool over STDIO or via the MCP Gateway's HTTP proxy:

### STDIO Example (direct)

```bash
# Launch a CLI client attached to the job container
docker exec -i avatar-renderer-container /app/.venv/bin/python \
  /app/mcp_client.py <<EOF
{
  "tool": "render_avatar",
  "params": {
    "avatar_path": "models/alice.png",
    "audio_path": "audio/hello.wav",
    "driver_video": null
  }
}
EOF
```

The server will reply with JSON:

```json
{ "jobId": "1234-abcd", "output": "/tmp/1234-abcd.mp4" }
```

### HTTP Proxy Example

If your MCP Gateway is configured with an HTTP proxy transport, you can call:

```bash
curl -X POST http://<GATEWAY_HOST>/api/tools/avatar_renderer/render_avatar \
  -H "Authorization: Bearer <USER_JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "avatar_path": "models/alice.png",
    "audio_path": "audio/hello.wav"
  }'
```

The gateway will forward the request to the STDIO server and return the JSON reply.

---

## 6 · Health Checking & Monitoring

* **Liveness Probe**: `scripts/healthcheck.sh` returns HTTP 200 if both STDIO and REST servers are healthy.
* **Metrics**: Export Prometheus metrics from `pipeline.py` via a `/metrics` endpoint if needed.

---

## 7 · Troubleshooting

* **Tool not shown**: Verify `autoDiscover: true` in `mcp-tool.json` and correct paths in `command` / `args`.
* **STDIO errors**: Check container logs (`kubectl logs`) for parse errors or missing dependencies.
* **Permission issues**: Ensure the container user can execute the Python binary and read model files.

---

## 8 · Cleanup

To remove the server from MCP Gateway:

```bash
curl -X DELETE http://<GATEWAY_HOST>/api/servers/avatar_renderer \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

Then scale down or delete the Kubernetes deployment:

```bash
helm uninstall avatar-renderer -n videogenie
```

---

*End of MCP Integration Guide.*
