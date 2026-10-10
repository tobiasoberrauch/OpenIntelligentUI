import { describe, expect, it } from "vitest";
import { readProviderHeaders } from "../provider-keys";

describe("request-scoped provider credentials", () => {
  it("leaves server mode unchanged when no key is supplied", () => {
    expect(readProviderHeaders(new Headers())).toBeUndefined();
  });
  it("forwards only the trimmed OpenAI key", () => {
    expect(readProviderHeaders(new Headers({
      "x-openai-api-key": "  openai-secret  ",
      "x-jev-api-key": "legacy-secret",
      authorization: "unrelated-secret",
    }))).toEqual({ "x-openai-api-key": "openai-secret" });
  });
  it.each(["", "has space", "x".repeat(4097)])("rejects invalid keys without echoing them", (key) => {
    expect(() => readProviderHeaders(new Headers({"x-openai-api-key": key}))).toThrow("Enter an OpenAI API key");
  });
});
