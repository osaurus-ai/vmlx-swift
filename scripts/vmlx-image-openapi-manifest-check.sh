#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT}/docs/OSAURUS_IMAGE_UI_MANIFEST.json"
OPENAPI="${ROOT}/docs/OSAURUS_IMAGE_OPENAPI.json"
REQUIRE_LOCAL_PROOF="${VMLX_REQUIRE_LOCAL_PROOF:-0}"

jq empty "$MANIFEST" >/dev/null
jq empty "$OPENAPI" >/dev/null

node - "$MANIFEST" "$OPENAPI" "$REQUIRE_LOCAL_PROOF" <<'NODE'
const fs = require("fs");
const path = require("path");

const [manifestPath, openapiPath, requireLocalProofRaw] = process.argv.slice(2);
const root = path.dirname(path.dirname(manifestPath));
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const openapi = JSON.parse(fs.readFileSync(openapiPath, "utf8"));
const requireLocalProof = requireLocalProofRaw === "1";

const failures = [];
const check = (condition, message) => {
  if (!condition) failures.push(message);
};

const expectedRoutes = {
  models: "GET /v1/images/models",
  generate: "POST /v1/images/generations",
  edit: "POST /v1/images/edits",
  upscale: "POST /v1/images/upscale",
  cancel: "POST /v1/images/cancel",
  job_status: "GET /v1/images/jobs/{id}",
  job_events: "GET /v1/images/jobs/{id}/events",
  output: "GET /v1/images/{image_id}",
};

check(manifest.status?.overall === "PARTIAL", "manifest status must remain PARTIAL until Osaurus HTTP/UI live proof exists");
check(openapi.info?.["x-vmlx-status"] === manifest.status?.overall, "OpenAPI x-vmlx-status must match manifest status");
check(openapi.info?.["x-vmlx-manifest"] === "docs/OSAURUS_IMAGE_UI_MANIFEST.json", "OpenAPI must point to the UI manifest");
check(manifest.source_trace?.current_proof_runner === "scripts/vmlx-image-current-proof.sh", "manifest must point to the current proof runner");

for (const [key, expected] of Object.entries(expectedRoutes)) {
  const actual = manifest.http_surface?.[key];
  check(actual === expected, `manifest http_surface.${key} expected ${expected}, got ${actual}`);
  const [method, routePath] = expected.split(" ");
  const pathItem = openapi.paths?.[routePath];
  check(pathItem, `OpenAPI missing path ${routePath}`);
  check(pathItem?.[method.toLowerCase()], `OpenAPI missing ${method} operation for ${routePath}`);
}

for (const routePath of ["/v1/images/generations", "/v1/images/edits", "/v1/images/upscale"]) {
  check(
    openapi.paths?.[routePath]?.post?.["x-vmlx-metal-gate-required"] === true,
    `${routePath} must declare x-vmlx-metal-gate-required=true`
  );
}

const schemas = openapi.components?.schemas ?? {};
for (const schema of [
  "ImageModel",
  "ImageCapabilities",
  "ImageModelDefaults",
  "ImageModelLimits",
  "ImageGenerationRequest",
  "ImageEditRequest",
  "ImageUpscaleRequest",
  "ImageJob",
  "ImageEvent",
  "ImageError",
]) {
  check(Boolean(schemas[schema]), `OpenAPI missing schema ${schema}`);
}

const qwenEdit = manifest.models?.find((model) => model.canonical === "qwen-image-edit");
check(Boolean(qwenEdit), "manifest missing qwen-image-edit model");
if (qwenEdit) {
  const shownIds = new Set((qwenEdit.variants ?? []).map((variant) => variant.model_id));
  check(shownIds.has("Qwen-Image-Edit-mflux-q4"), "qwen edit q4 must remain exposed");
  check(shownIds.has("Qwen-Image-Edit-mflux-q5"), "qwen edit q5 must remain exposed");
  check(shownIds.has("Qwen-Image-Edit-mflux-q8"), "qwen edit q8 must remain exposed");
  check(!shownIds.has("Qwen-Image-Edit-mflux-q3"), "qwen edit q3 must not be exposed as a normal variant");
  check((qwenEdit.hide_controls ?? []).includes("mask"), "qwen edit must hide mask controls");
  check((qwenEdit.controls ?? []).includes("source_images"), "qwen edit must expose ordered source_images");
  const blockedIds = new Set((qwenEdit.blocked_variants ?? []).map((variant) => variant.model_id));
  check(blockedIds.has("Qwen-Image-Edit-mflux-q3"), "qwen edit q3 boundary must be listed as blocked");
  const q4 = (qwenEdit.variants ?? []).find((variant) => variant.model_id === "Qwen-Image-Edit-mflux-q4");
  const q5 = (qwenEdit.variants ?? []).find((variant) => variant.model_id === "Qwen-Image-Edit-mflux-q5");
  const q8 = (qwenEdit.variants ?? []).find((variant) => variant.model_id === "Qwen-Image-Edit-mflux-q8");
  check((q4?.proof_status ?? "").includes("20260618"), "qwen edit q4 proof status must mention 20260618 stress proof");
  check((q5?.proof_status ?? "").includes("20260618"), "qwen edit q5 proof status must mention 20260618 stress proof");
  check((q8?.proof_status ?? "").includes("20260618"), "qwen edit q8 proof status must mention 20260618 stress proof");
}

const ideogram = manifest.models?.find((model) => model.canonical === "ideogram");
check(Boolean(ideogram), "manifest missing ideogram model");
if (ideogram) {
  const ids = new Set((ideogram.variants ?? []).map((variant) => variant.model_id));
  check(
    ideogram.ui_exposure === "show_for_testing_with_json_caption_caveat",
    "ideogram must keep JSON-caption caveat UI exposure"
  );
  check(ids.has("ideogram-4-fp8"), "ideogram fp8 staged mirror variant missing");
  check(ids.has("ideogram-4-nf4"), "ideogram NF4 staged mirror variant missing");
  check((ideogram.hide_controls ?? []).includes("mask"), "ideogram must hide mask controls");
}

function collectStrings(value, out = []) {
  if (typeof value === "string") out.push(value);
  else if (Array.isArray(value)) value.forEach((item) => collectStrings(item, out));
  else if (value && typeof value === "object") Object.values(value).forEach((item) => collectStrings(item, out));
  return out;
}

function collectSha(value, out = new Set()) {
  if (Array.isArray(value)) value.forEach((item) => collectSha(item, out));
  else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if ((key === "sha256" || key === "sha") && typeof item === "string" && /^[0-9a-f]{64}$/i.test(item)) {
        out.add(item.toLowerCase());
      }
      if (key === "image_diagnostics" && item && typeof item === "object" && /^[0-9a-f]{64}$/i.test(item.sha256 ?? "")) {
        out.add(item.sha256.toLowerCase());
      }
      collectSha(item, out);
    }
  }
  return out;
}

let proofObjects = [];
function walk(value, visit) {
  if (Array.isArray(value)) return value.forEach((item) => walk(item, visit));
  if (value && typeof value === "object") {
    visit(value);
    Object.values(value).forEach((item) => walk(item, visit));
  }
}
walk(manifest, (object) => {
  if (typeof object.proof_artifact === "string") proofObjects.push(object);
});

let localProofStatus = "skipped";
if (requireLocalProof) {
  localProofStatus = "checked";
  for (const object of proofObjects) {
    const artifactPath = path.join(root, object.proof_artifact);
    if (!fs.existsSync(artifactPath)) {
      failures.push(`${object.proof_artifact}: missing proof artifact`);
      continue;
    }
    const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
    const artifactShas = collectSha(artifact);
    const expectedShas = collectStrings(object)
      .filter((text) => /^[0-9a-f]{64}$/i.test(text))
      .map((text) => text.toLowerCase());
    for (const sha of expectedShas) {
      check(artifactShas.has(sha), `${object.proof_artifact}: expected sha ${sha} not found in artifact`);
    }
  }
  for (const key of ["status_load_matrix", "visual_contact_sheet", "current_proof_summary", "current_proof_contact_sheet"]) {
    const relative = manifest.runtime_evidence?.[key];
    check(Boolean(relative), `runtime_evidence.${key} missing`);
    if (relative) check(fs.existsSync(path.join(root, relative)), `runtime_evidence.${key} missing on disk: ${relative}`);
  }

  const summaryRelative = manifest.runtime_evidence?.current_proof_summary;
  if (summaryRelative && fs.existsSync(path.join(root, summaryRelative))) {
    const summary = JSON.parse(fs.readFileSync(path.join(root, summaryRelative), "utf8"));
    check(summary.status === "passed", "current proof summary status must be passed");
    check(summary.matrix?.model_count === 14, "current proof summary matrix model_count must be 14");
    check(summary.matrix?.loaded_count === 14, "current proof summary matrix loaded_count must be 14");
    const expectedProofRows = new Set([
      "zimage-4bit-gen",
      "zimage-8bit-gen",
      "flux-schnell-4bit-gen",
      "flux-schnell-8bit-gen",
      "qwen-image-4bit-gen",
      "qwen-image-8bit-gen",
      "qwen-edit-q4-gen",
      "qwen-edit-q8-gen",
      "ideogram-fp8-gen",
      "ideogram-nf4-gen",
    ]);
    const rows = Array.isArray(summary.rows) ? summary.rows : [];
    const actualRows = new Set(rows.map((row) => row.label));
    for (const label of expectedProofRows) {
      check(actualRows.has(label), `current proof summary missing row ${label}`);
    }
    for (const row of rows) {
      if (!expectedProofRows.has(row.label)) continue;
      check(row.status === "passed", `current proof summary row ${row.label} must be passed`);
      check(row.deterministic_repeat === true, `current proof summary row ${row.label} missing deterministic repeat`);
      check(row.prompt_sensitive === true, `current proof summary row ${row.label} missing prompt sensitivity`);
      check(Array.isArray(row.shas) && row.shas.length === 3, `current proof summary row ${row.label} must include three shas`);
      check(Array.isArray(row.outputs) && row.outputs.length === 3, `current proof summary row ${row.label} must include three outputs`);
    }
  }
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log(JSON.stringify({
  manifest: path.relative(root, manifestPath),
  openapi: path.relative(root, openapiPath),
  routes: Object.keys(expectedRoutes).length,
  proofObjects: proofObjects.length,
  localProofStatus,
}, null, 2));
NODE
