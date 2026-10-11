# Stage 19 — MaSIF method and application references

## Purpose and intended work

Complete the README's references for the integrated MaSIF, MaSIF-seed and
MaSIF-neosurface components, and include directly related MaSIF research.
Provide consistent, reusable BibTeX records and identify which papers correspond
to implemented applications. Verify publication metadata against original sources.

## Implemented changes

- `README.md`: replace the single citation with three original method papers and
  four related-method/application references, including linked papers and the
  correspondence to application names.
- `citation.bib`: retain `gainza2020deciphering` and add six records with complete
  author lists, verified venues/years, available volume/page information and
  DOI or official paper URLs.
- `docs/USAGE.md`: synchronize the same reference list and link the BibTeX file.
- `docs/torch_migration/REPORT.md`: link this documentation stage.

The three original method references cover MaSIF-site/search/ligand and scoring,
MaSIF-seed, and MaSIF-neosurf. Related reading covers dMaSIF, physics-informed
rigid-body docking, DiffMaSIF and the human surfaceome/SURFACE-Bind application.
Their inclusion does not claim implementation of these additional architectures
or applications. This list covers the original integrated methods and identified
direct family extensions; it is not an inventory of every paper citing MaSIF.

## Sources and bibliographic decisions

- [MaSIF DOI](https://doi.org/10.1038/s41592-019-0666-6): use the 2020 journal issue,
  Nature Methods17,184–192; online publication was December2019.
- [MaSIF-seed Nature publication](https://www.nature.com/articles/s41586-023-05993-x):
  use the final title, without the preprint's additional “site-specific”, and
  Nature617,176–184(2023).
- [MaSIF-neosurf Nature publication](https://www.nature.com/articles/s41586-024-08435-4):
  Nature639,522–531(2025).
- [dMaSIF official CVF paper and BibTeX](https://openaccess.thecvf.com/content/CVPR2021/html/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.html):
  follow CVF's own15272–15281 pagination and URL. Crossref's IEEE DOI record lists
  a different pagination; do not combine that DOI with the CVF page numbers.
- [Physics-informed docking author record](https://www.jeanfeydy.com/research.html)
  and [second author record](https://jks17.github.io/) verify the author order and
  MLDD at ICLR2022. Link the original
  [OpenReview record](https://openreview.net/forum?id=5yn5shS6wN). Direct OpenReview
  and HAL requests encountered access challenges; no unverified DOI/pages are added.
- [DiffMaSIF official workshop paper](https://www.mlsb.io/papers_2023/DiffMaSIF_Surface-based_Protein-Protein_Docking_with_Diffusion_Models.pdf):
  cite the MLSB workshop at NeurIPS2023, not an accepted ICLR main-conference paper.
- [Surfaceome application PNAS publication](https://www.pnas.org/doi/10.1073/pnas.2506269123):
  use the 2026 version of record,123(5),e2506269123, rather than an earlier preprint.

Author lists and journal metadata for the four journal papers are checked against
publisher-deposited Crossref DOI records. An independent subagent verifies scope,
publication information and consistency of the final reference additions.

## Verification and numerical scope

This stage edits documentation and bibliography only. Neural weights, model
implementation, training, preprocessing and task protocols are unchanged.

- Validate seven unique BibTeX keys, balanced entries, required fields and matching
  paper links in both README and detailed guide.
- Run the existing smallest numerical regression suite:
  `python -m unittest discover -s tests -p test_numerics.py -v`.
- Retain the existing TF/Torch precision limits; documentation additions introduce
  no new numerical equivalence or performance claim.

## Final results

Seven unique BibTeX entries pass balanced-brace and required-field checks, with
matching paper URLs in both README and USAGE. Independent review confirms author
order, published titles, venues, years and the implemented/related scope. PNAS's
article identifier is stored in `pages` for compatibility with classic BibTeX
styles; no standalone bibliography rendering is claimed.

The existing numerical suite passes all **11 tests without skips**. No model,
checkpoint or runtime source was edited in this stage.
