# Local Artifact Archive

On 2026-08-04, local-only HAR-V run downloads, raw prediction panels, Actions
artifacts, smoke probes, caches, and temporary copies were moved off the system
drive. Canonical source, configuration, tests, cleaned summaries, selected
tables and figures, final-paper inputs, and approved tracked results remain in
Git.

Archive location:

```text
D:\Codex_Offloaded_Data\money-making\local-artifacts-2026-08-04
```

The archive contains:

- `manifest.csv`: relative path, byte count, and SHA-256 for every archived file.
- `verification.json`: archive totals and verification result.
- `payload/`: the byte-for-byte verified archive tree.
- `source-originals/`: the original local-only directories moved from C: after
  verification.

Verification covered 1,046 files totaling 2,582,051,591 bytes and reported zero
mismatches. The SHA-256 of `manifest.csv` is
`62FAFF3DA2E21D53E4D760D5CEECC97809BF604B873FD0FA69DACF5E73D82C12`.

To restore an artifact, find its `RelativePath` in `manifest.csv` and copy the
corresponding file from `payload/` back under the repository root. No junctions
are required: the archived paths are generated/downloaded output locations,
and the tracked runners recreate them when needed.
