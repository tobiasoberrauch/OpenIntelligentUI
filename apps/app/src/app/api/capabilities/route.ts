import { normalizeLanggraphUrl } from "@/lib/copilotkit-runtime-options";

// Fail closed: an unreachable or unconfigured agent means visitors must enter keys.
export async function GET() {
  try {
    const base = normalizeLanggraphUrl(process.env.LANGGRAPH_DEPLOYMENT_URL);
    const response = await fetch(`${base.replace(/\/$/, "")}/capabilities`, {
      cache: "no-store",
      redirect: "error",
      signal: AbortSignal.timeout(5_000),
    });
    if (!response.ok) throw new Error("agent capabilities unavailable");
    const data = (await response.json()) as { server_credentials?: unknown };
    return Response.json({ server_credentials: data.server_credentials === true }, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json({ server_credentials: false }, {
      status: 502,
      headers: { "Cache-Control": "no-store" },
    });
  }
}
