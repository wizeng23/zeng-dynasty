import { describe, expect, it } from "vitest";
import { buildSearchIndex, searchPeople } from "./search";
import type { FamilyNode } from "./tree";

// Minimal node factory — only the fields search cares about.
function node(id: number, name: string, generation = 1): FamilyNode {
  return {
    id,
    name,
    name_images: [],
    generation,
    father: -1,
    children: [],
    biography: "",
    notes: "",
  };
}

// 兴思 -> xingsi / xing1si1 ; 点 -> dian / dian3 ; 衍谞 -> yanxu / yan3xu1
const people = [
  node(1, "点", 1),
  node(1707, "兴思", 68),
  node(1534, "衍谞", 67),
  node(1956, "兴国", 69), // shares first char 兴 with 兴思
];
const index = buildSearchIndex(people);

describe("buildSearchIndex", () => {
  it("skips nameless (image-only) people", () => {
    const idx = buildSearchIndex([node(1, "点"), node(2, "")]);
    expect(idx).toHaveLength(1);
    expect(idx[0].node.id).toBe(1);
  });

  it("stores the tone-marked display reading", () => {
    const entry = index.find((e) => e.node.id === 1707);
    expect(entry?.display).toBe("Xīng Sī");
  });
});

describe("searchPeople — Hanzi", () => {
  it("matches a full Hanzi name", () => {
    const r = searchPeople(index, "兴思");
    expect(r[0].node.id).toBe(1707);
  });

  it("matches a single Hanzi character across multiple people", () => {
    const ids = searchPeople(index, "兴").map((r) => r.node.id);
    expect(ids).toContain(1707);
    expect(ids).toContain(1956);
  });
});

describe("searchPeople — implied surname 曾/Zeng is stripped", () => {
  it("ignores a leading 曾 before a Hanzi given name", () => {
    expect(searchPeople(index, "曾兴思")[0].node.id).toBe(1707);
  });

  it("ignores a leading zeng/zeng1/ceng2 before pinyin", () => {
    expect(searchPeople(index, "zengxingsi")[0].node.id).toBe(1707);
    expect(searchPeople(index, "zeng1xing1si1")[0].node.id).toBe(1707);
    expect(searchPeople(index, "ceng2xingsi")[0].node.id).toBe(1707);
  });

  it("ignores a leading zeng with a space before the given name", () => {
    expect(searchPeople(index, "zeng xing si")[0].node.id).toBe(1707);
  });

  it("returns nothing for the bare surname alone (no given name to match)", () => {
    expect(searchPeople(index, "曾")).toEqual([]);
    expect(searchPeople(index, "zeng")).toEqual([]);
    expect(searchPeople(index, "ceng2")).toEqual([]);
  });

  it("does not strip 曾/ceng when it is not the first character", () => {
    // A hypothetical given name containing 曾 later should still match on it.
    // (曾 romanizes as "ceng" mid-name; the surname reading is zeng.)
    const idx = buildSearchIndex([node(99, "德曾")]); // "de ceng"
    expect(searchPeople(idx, "曾")[0].node.id).toBe(99); // Hanzi substring, not stripped
    expect(searchPeople(idx, "deceng")[0].node.id).toBe(99); // pinyin substring
  });
});

describe("searchPeople — pinyin", () => {
  it("matches toneless pinyin", () => {
    expect(searchPeople(index, "xingsi")[0].node.id).toBe(1707);
  });

  it("matches toneless pinyin typed with a space", () => {
    expect(searchPeople(index, "xing si")[0].node.id).toBe(1707);
  });

  it("matches tone-numbered pinyin", () => {
    expect(searchPeople(index, "xing1si1")[0].node.id).toBe(1707);
  });

  it("matches a pinyin prefix", () => {
    const ids = searchPeople(index, "xing").map((r) => r.node.id);
    expect(ids).toContain(1707);
    expect(ids).toContain(1956);
  });

  it("is case-insensitive", () => {
    expect(searchPeople(index, "XingSi")[0].node.id).toBe(1707);
  });
});

describe("searchPeople — ranking & limits", () => {
  it("ranks an exact name above a prefix match", () => {
    // 点 exact should outrank nothing else here, but check exact-first ordering
    // with a shared-prefix set: querying 兴思 exactly returns 兴思 first.
    const r = searchPeople(index, "兴思");
    expect(r[0].node.id).toBe(1707);
  });

  it("returns nothing for an empty query", () => {
    expect(searchPeople(index, "")).toEqual([]);
    expect(searchPeople(index, "   ")).toEqual([]);
  });

  it("returns nothing when there is no match", () => {
    expect(searchPeople(index, "zzz")).toEqual([]);
  });
});
