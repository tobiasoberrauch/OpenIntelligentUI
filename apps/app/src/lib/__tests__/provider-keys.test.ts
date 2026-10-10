import { describe, expect, it } from "vitest";
import { readProviderHeaders } from "../provider-keys";

describe("request-scoped provider credentials", () => {
  it("leaves server mode unchanged when no keys are supplied", () => {
    expect(readProviderHeaders(new Headers())).toBeUndefined();
  });
  it("forwards only the trimmed OpenAI key", () => {
    expect(readProviderHeaders(new Headers({
      "x-openai-api-key": "  openai-secret  ",
      authorization: "unrelated-secret",
    }))).toEqual({ "x-openai-api-key": "openai-secret" });
  });
  it("forwards an optional trimmed Jev key", () => {
    expect(readProviderHeaders(new Headers({
      "x-openai-api-key": "openai-secret",
      "x-jev-api-key": "  jev-secret ",
    }))).toEqual({ "x-openai-api-key": "openai-secret", "x-jev-api-key": "jev-secret" });
  });
  it.each(["", "has space", "x".repeat(4097)])("rejects invalid keys without echoing them", (key) => {
    expect(() => readProviderHeaders(new Headers({"x-openai-api-key": key}))).toThrow("Enter an OpenAI API key");
    expect(() => readProviderHeaders(new Headers({"x-openai-api-key": "ok", "x-jev-api-key": key}))).toThrow("Enter an OpenAI API key");
  });
  it("rejects a Jev key without an OpenAI key", () => {
    expect(() => readProviderHeaders(new Headers({"x-jev-api-key": "jev-secret"}))).toThrow("Enter an OpenAI API key");
  });
});
