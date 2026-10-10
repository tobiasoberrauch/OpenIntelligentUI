// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";
import { GET } from "./route";

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

describe("capabilities proxy", () => {
  it("reports configured server credentials from the agent", async () => {
    vi.stubEnv("LANGGRAPH_DEPLOYMENT_URL", "agent:8123");
    const fetchMock = vi.fn().mockResolvedValue(Response.json({ server_credentials: true }));
    vi.stubGlobal("fetch", fetchMock);
    const response = await GET();
    expect(fetchMock).toHaveBeenCalledWith("http://agent:8123/capabilities", expect.objectContaining({ cache: "no-store", redirect: "error" }));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ server_credentials: true });
    expect(response.headers.get("cache-control")).toBe("no-store");
  });

  it("fails closed when the agent is unreachable or reports nothing", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("agent down")));
    let response = await GET();
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ server_credentials: false });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ server_credentials: "yes" })));
    response = await GET();
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ server_credentials: false });
  });
});
