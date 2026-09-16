import { describe, expect, it } from "vitest";
import { toDisplayPinyin, toSearchPinyin } from "./pinyin";

describe("toDisplayPinyin", () => {
  it("renders tone marks, per-syllable capitalized, space-separated", () => {
    expect(toDisplayPinyin("兴思")).toBe("Xīng Sī");
  });

  it("handles a single character", () => {
    expect(toDisplayPinyin("点")).toBe("Diǎn");
  });

  it("returns empty for empty or whitespace input", () => {
    expect(toDisplayPinyin("")).toBe("");
    expect(toDisplayPinyin("   ")).toBe("");
  });
});

describe("toSearchPinyin", () => {
  it("produces toneless and tone-numbered forms, lowercased, no spaces", () => {
    expect(toSearchPinyin("兴思")).toEqual({
      toneless: "xingsi",
      toneNumbered: "xing1si1",
    });
  });

  it("handles a single character", () => {
    expect(toSearchPinyin("点")).toEqual({
      toneless: "dian",
      toneNumbered: "dian3",
    });
  });

  it("returns empty forms for empty input", () => {
    expect(toSearchPinyin("")).toEqual({ toneless: "", toneNumbered: "" });
  });
});
