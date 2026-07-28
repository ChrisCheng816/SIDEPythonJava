import copy
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
APP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"

ET.register_namespace("a", A_NS)
ET.register_namespace("p", P_NS)
ET.register_namespace("r", R_NS)


def qname(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def build_text_body(paragraphs: list[dict], bullet_mode: bool) -> ET.Element:
    tx_body = ET.Element(qname(P_NS, "txBody"))
    body_pr = ET.SubElement(tx_body, qname(A_NS, "bodyPr"))
    body_pr.set("rtlCol", "0")
    if bullet_mode:
        body_pr.set("anchor", "t")
    ET.SubElement(tx_body, qname(A_NS, "lstStyle"))

    for paragraph in paragraphs:
        p = ET.SubElement(tx_body, qname(A_NS, "p"))
        text = paragraph["text"]
        level = paragraph.get("level", 0)
        if bullet_mode:
            p_pr = ET.SubElement(p, qname(A_NS, "pPr"))
            p_pr.set("lvl", str(level))

        run = ET.SubElement(p, qname(A_NS, "r"))
        r_pr = ET.SubElement(run, qname(A_NS, "rPr"))
        r_pr.set("lang", "en-US")
        if "size" in paragraph:
            r_pr.set("sz", str(paragraph["size"]))
        if paragraph.get("bold"):
            r_pr.set("b", "1")
        if "color" in paragraph:
            solid_fill = ET.SubElement(r_pr, qname(A_NS, "solidFill"))
            srgb = ET.SubElement(solid_fill, qname(A_NS, "srgbClr"))
            srgb.set("val", paragraph["color"])

        t = ET.SubElement(run, qname(A_NS, "t"))
        t.text = text

        end = ET.SubElement(p, qname(A_NS, "endParaRPr"))
        end.set("lang", "en-US")
        if "size" in paragraph:
            end.set("sz", str(paragraph["size"]))
        if "color" in paragraph:
            solid_fill = ET.SubElement(end, qname(A_NS, "solidFill"))
            srgb = ET.SubElement(solid_fill, qname(A_NS, "srgbClr"))
            srgb.set("val", paragraph["color"])

    if not paragraphs:
        p = ET.SubElement(tx_body, qname(A_NS, "p"))
        end = ET.SubElement(p, qname(A_NS, "endParaRPr"))
        end.set("lang", "en-US")

    return tx_body


def find_placeholder_shape(root: ET.Element, ph_type: str | None = None, idx: str | None = None) -> ET.Element | None:
    for sp in root.findall(f".//{qname(P_NS, 'sp')}"):
        ph = sp.find(f"./{qname(P_NS, 'nvSpPr')}/{qname(P_NS, 'nvPr')}/{qname(P_NS, 'ph')}")
        if ph is None:
            continue
        if ph_type is not None and ph.get("type") != ph_type:
            continue
        if idx is not None and ph.get("idx") != idx:
            continue
        return sp
    return None


def set_shape_text(sp: ET.Element, paragraphs: list[dict], bullet_mode: bool) -> None:
    tx_body = sp.find(f"./{qname(P_NS, 'txBody')}")
    if tx_body is not None:
        sp.remove(tx_body)
    sp.append(build_text_body(paragraphs, bullet_mode))


def write_xml(path: Path, root: ET.ElementTree | ET.Element) -> None:
    if isinstance(root, ET.Element):
        tree = ET.ElementTree(root)
    else:
        tree = root
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def slide_deck_definition() -> list[dict]:
    return [
        {
            "layout": "title",
            "title": "Reproducing SIDE and Extending It to Python",
            "subtitle": [
                "Paper focus: code-summary semantic alignment beyond reference overlap",
                "Java base artifact: /scratch/zcheng06/code-summarization-metric",
                "Python extension: /scratch/zcheng06/SidePython",
                "March 9, 2026",
            ],
        },
        {
            "layout": "content",
            "title": "Research Problem and Motivation",
            "bullets": [
                "Reference-based metrics can underrate summaries that are semantically correct but lexically different.",
                "Repository comments may be stale, incomplete, or low quality, so reference text is not always a reliable gold standard.",
                "The selected paper asks whether a code-aware semantic metric can better match developer judgment.",
                "SIDE measures alignment between the code snippet and the generated summary, not only overlap with a reference comment.",
                "This matters for fair evaluation of code summarizers and for data-centric filtering decisions.",
            ],
        },
        {
            "layout": "content",
            "title": "Overview of the Original Approach",
            "bullets": [
                "Build triplets: query = code, positive = suitable summary, negative = unsuitable summary.",
                "Encode texts with MPNet plus mean pooling and optimize a triplet-loss embedding space.",
                "At inference time, SIDE is the cosine similarity between the code embedding and the summary embedding.",
                "Model selection uses the average margin cos(query,pos) - cos(query,neg) on held-out evaluation triplets.",
                "The artifact also benchmarks SIDE against lexical and embedding-based baselines.",
            ],
        },
        {
            "layout": "content",
            "title": "Data, Benchmarks, and Main Results",
            "bullets": [
                "Java training/eval triplets and pretrained checkpoints are distributed separately from the repo via external links.",
                "The human benchmark CSV contains 6,253 annotated code-summary pairs; 5,201 remain after filtering mid != 0.",
                "Compared metrics include Jaccard, BLEU, ROUGE, METEOR, USE, BERTScore, SentenceBERT, InferSent, and CodeT5+.",
                "Local Java checkpoint selection reproduces the best checkpoint at 141205 with eval margin 0.81698.",
                "The reproduced regression analysis retains SIDE in the reduced metric set and shows significance for all judged quality dimensions.",
            ],
        },
        {
            "layout": "content",
            "title": "Artifact and Reproducibility Assessment",
            "bullets": [
                "Strengths: the repo includes training code, evaluation scripts, R analysis, precomputed CSVs, and paper-aligned logic.",
                "Weak points in the original Java clone: missing dataset/model folders, hard-coded paths, CUDA-only defaults, and incomplete dependency notes.",
                "README relies on Google Drive assets for the actual triplets and trained checkpoints.",
                "The original Analysis script assumed a repo named summarization-metric and a CSV at repo root, so it failed out of the box.",
                "Result: the artifact was close to reproducible, but not directly runnable in a fresh local clone.",
            ],
        },
        {
            "layout": "content",
            "title": "Installation, Dependency, and Usability Challenges",
            "bullets": [
                "Java requirements do not cover every metric script; computeAllMetrics references extra packages such as sacrebleu, rouge, bert_score, and tensorflow_hub.",
                "R analysis depends on MASS, Hmisc, and xtable; Hmisc and xtable are R packages, not pip packages.",
                "SidePython audit found 39 JSONL Git LFS pointer files and 34 scripts with missing modules under bare python3.",
                "Several legacy scripts still assume machine-specific or repository-specific paths.",
                "Environment fragmentation is real: the Java repo uses a local uv/venv setup, while SidePython provides a Conda environment.",
            ],
        },
        {
            "layout": "content",
            "title": "Reproduction Progress: Recovering Java SIDE",
            "bullets": [
                "Added the local assets expected by the Java project: fine-tuning/fine-tuning/train.json, eval.json, and hard-negatives/hard-negatives/141205.",
                "Reworked Analysis/Analysis-SIDE.r to auto-detect the repo root, benchmark CSV, datasets, and checkpoint folder.",
                "Added dependency checks and fallback behavior when Hmisc or xtable are missing.",
                "Added Analysis/README.md and ignored generated outputs so reruns are cleaner.",
                "Current command Rscript Analysis/Analysis-SIDE.r now completes successfully in this workspace.",
            ],
        },
        {
            "layout": "content",
            "title": "Reproduced Results: Java Analysis",
            "bullets": [
                "Best Java checkpoint from Results/evaluation/evalResults.csv: 141205, cosine margin 0.81698.",
                "Reproduced regression on Overall DA Score: SIDE coefficient = 0.0192, OR = 1.0194, p < 0.001.",
                "SIDE is also significant for Content Adequacy (0.4535), Conciseness (0.2973), Fluency (0.2348), and composite Overall (0.3544), all p < 0.001.",
                "In Spearman correlation, SIDE is most associated with USE_CS (0.411), SentenceBERT_CS (0.405), and c_coeff (0.408), but remains an independent retained metric.",
                "This supports the paper's motivation: code-summary semantic alignment adds signal beyond token overlap alone.",
            ],
        },
        {
            "layout": "content",
            "title": "Planned Extensions",
            "bullets": [
                "Retrain SIDE for Python code-summary pairs instead of reusing the Java model unchanged.",
                "Evaluate SIDE-py on Python benchmark predictions from baseline, AST, Function Signature, and CrystalBLEU data-reduction variants.",
                "Compare SIDE-py against BLEU-4, ROUGE-L, METEOR, ChrF, and TF-IDF cosine on the Python benchmark.",
                "Extend the reproduction from Java-only artifact repair to cross-language empirical validation.",
                "If time permits, add hard-negative variants, stronger validation splits, and more ablation settings.",
            ],
        },
        {
            "layout": "content",
            "title": "SidePython Artifact Overview",
            "bullets": [
                "SidePython organizes the extension into three studies: Java token optimization, SIDE-py training, and Python generalization.",
                "study-2/training-sidep contains the MPNet triplet training code and the current SIDE-py checkpoint.",
                "study-3/inference-results stores Python benchmark predictions, while pipeline/ adds staged entrypoints for prep, train, score, and compare.",
                "run_all.py performs preflight checks for missing modules and Git LFS placeholders before running the staged workflow.",
                "environment.yml is much more complete than the original Java requirements and serves as the intended reproducible environment entrypoint.",
            ],
        },
        {
            "layout": "content",
            "title": "Current Python Progress",
            "bullets": [
                "Prepared 3,600 training triplets and 400 validation triplets for SIDE-py.",
                "Current checkpoint: study-2/training-sidep/models/mpnet_triplet_no_hardneg_v2-test.",
                "Validation score improved from 0.4634 at epoch 1 to a best 0.6003 at epoch 9.",
                "Scored 500 Python benchmark predictions; SIDE shows moderate correlation with ChrF (0.4363), METEOR (0.4360), TF-IDF cosine (0.4361), and ROUGE-L (0.4146).",
                "The remaining gap is the full human-judgment statistical replication for Python.",
            ],
        },
        {
            "layout": "content",
            "title": "Artifact Improvements",
            "bullets": [
                "Java: fixed the non-runnable R analysis by removing hard-coded paths and adding robust asset discovery.",
                "Java: added README guidance and stable output handling for repeatable reruns.",
                "Python: introduced wrapper scripts run_train_sidepy.py, run_score_sidepy.py, run_compare_metrics.py, and run_all.py.",
                "Python: added preflight auditing to surface Git LFS placeholders and missing modules early.",
                "Planned next improvements: containerization, smoke tests, and a unified top-level reproduction guide.",
            ],
        },
        {
            "layout": "content",
            "title": "Solo Contributions and Workflow",
            "bullets": [
                "Reproduced and repaired the original Java SIDE artifact in the local clone.",
                "Added the missing local assets and validated the end-to-end Analysis output.",
                "Built the SidePython pipeline wrappers and trained the current SIDE-py checkpoint.",
                "Generated benchmark-side metric summaries and audit reports to document current artifact status.",
                "Workflow is currently solo and commit-centric; the next process upgrade is to formalize GitHub issues, milestones, and PR-style review for the remaining work.",
            ],
        },
        {
            "layout": "content",
            "title": "Challenges and Risks",
            "bullets": [
                "Some critical datasets are still external or stored as Git LFS pointers, so cloning alone is not sufficient.",
                "Cross-repo dependency management is inconsistent, especially across Python and R environments.",
                "Python extension quality currently relies on automatic metrics; human annotation and statistical validation remain the main experimental risk.",
                "Several legacy scripts contain machine-specific paths, which raises portability and onboarding cost.",
                "If GPU availability or package resolution changes, training and scoring throughput could become a bottleneck.",
            ],
        },
        {
            "layout": "content",
            "title": "Next Steps",
            "bullets": [
                "Complete the Python-side reproduction loop: score all benchmark variants and run the full comparison pipeline.",
                "Port the Java-style statistical analysis to the Python benchmark with SIDE-py included as a metric.",
                "Clean up LFS/data acquisition instructions and add a single start-to-finish reproduction document.",
                "Separate active development into milestone-sized branches or tasks and stabilize the final artifact.",
                "Prepare the final report with a clear distinction between reproduced results and true extension results.",
            ],
        },
        {
            "layout": "content",
            "title": "Lessons Learned (so far)",
            "bullets": [
                "Reproducibility failures are often mundane: paths, hidden external assets, and under-specified environments matter as much as model code.",
                "A small amount of automation dramatically improves artifact usability.",
                "Semantic metrics like SIDE matter because they test a different assumption than reference-overlap metrics.",
                "Extension work should start only after the base artifact is executable and auditable.",
                "For empirical software engineering, artifact quality is part of the scientific contribution, not just packaging.",
            ],
        },
    ]


def populate_title_slide(slide_path: Path, title: str, subtitle_lines: list[str]) -> None:
    tree = ET.parse(slide_path)
    root = tree.getroot()
    title_sp = find_placeholder_shape(root, ph_type="ctrTitle")
    subtitle_sp = find_placeholder_shape(root, ph_type="subTitle")
    if title_sp is None or subtitle_sp is None:
        raise RuntimeError("Could not find title placeholders in template title slide.")

    set_shape_text(
        title_sp,
        [{"text": title, "size": 3000, "bold": True, "color": "FFFFFF"}],
        bullet_mode=False,
    )
    set_shape_text(
        subtitle_sp,
        [{"text": line, "size": 1800, "color": "FFFFFF"} for line in subtitle_lines],
        bullet_mode=False,
    )
    write_xml(slide_path, tree)


def populate_content_slide(slide_path: Path, title: str, bullets: list[str]) -> None:
    tree = ET.parse(slide_path)
    root = tree.getroot()
    title_sp = find_placeholder_shape(root, ph_type="title")
    body_sp = find_placeholder_shape(root, idx="1")
    if title_sp is None or body_sp is None:
        raise RuntimeError(f"Could not find content placeholders in {slide_path}.")

    set_shape_text(
        title_sp,
        [{"text": title, "size": 2600, "bold": True}],
        bullet_mode=False,
    )
    set_shape_text(
        body_sp,
        [{"text": bullet, "size": 2000} for bullet in bullets],
        bullet_mode=True,
    )

    slide_num_sp = find_placeholder_shape(root, ph_type="sldNum")
    if slide_num_sp is not None:
        fld = slide_num_sp.find(f".//{qname(A_NS, 'fld')}")
        if fld is not None:
            fld.set("id", "{" + str(uuid.uuid4()).upper() + "}")

    write_xml(slide_path, tree)


def add_slide_relationship(rels_path: Path, rel_id: str, target: str) -> None:
    tree = ET.parse(rels_path)
    root = tree.getroot()
    rel = ET.SubElement(root, qname(PR_NS, "Relationship"))
    rel.set("Id", rel_id)
    rel.set(
        "Type",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
    )
    rel.set("Target", target)
    write_xml(rels_path, tree)


def add_slide_reference(presentation_path: Path, slide_id: int, rel_id: str) -> None:
    tree = ET.parse(presentation_path)
    root = tree.getroot()
    sld_id_lst = root.find(f"./{qname(P_NS, 'sldIdLst')}")
    if sld_id_lst is None:
        raise RuntimeError("presentation.xml missing p:sldIdLst")
    sld = ET.SubElement(sld_id_lst, qname(P_NS, "sldId"))
    sld.set("id", str(slide_id))
    sld.set(qname(R_NS, "id"), rel_id)
    write_xml(presentation_path, tree)


def add_content_type_override(content_types_path: Path, slide_number: int) -> None:
    tree = ET.parse(content_types_path)
    root = tree.getroot()
    override = ET.SubElement(root, qname(CT_NS, "Override"))
    override.set("PartName", f"/ppt/slides/slide{slide_number}.xml")
    override.set(
        "ContentType",
        "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
    )
    write_xml(content_types_path, tree)


def get_next_rel_id(rels_path: Path) -> int:
    tree = ET.parse(rels_path)
    root = tree.getroot()
    max_id = 0
    for rel in root.findall(f"./{qname(PR_NS, 'Relationship')}"):
        rel_id = rel.get("Id", "")
        if rel_id.startswith("rId"):
            try:
                max_id = max(max_id, int(rel_id[3:]))
            except ValueError:
                continue
    return max_id + 1


def get_next_slide_id(presentation_path: Path) -> int:
    tree = ET.parse(presentation_path)
    root = tree.getroot()
    sld_id_lst = root.find(f"./{qname(P_NS, 'sldIdLst')}")
    max_id = 255
    if sld_id_lst is not None:
        for sld in sld_id_lst.findall(f"./{qname(P_NS, 'sldId')}"):
            try:
                max_id = max(max_id, int(sld.get("id", "255")))
            except ValueError:
                continue
    return max_id + 1


def update_app_properties(app_path: Path, slide_titles: list[str]) -> None:
    tree = ET.parse(app_path)
    root = tree.getroot()

    slides_node = root.find(f"./{qname(APP_NS, 'Slides')}")
    if slides_node is not None:
        slides_node.text = str(len(slide_titles))

    heading_pairs = root.find(f"./{qname(APP_NS, 'HeadingPairs')}/{qname(VT_NS, 'vector')}")
    if heading_pairs is not None and len(list(heading_pairs)) >= 6:
        heading_pairs[-1].find(f"./{qname(VT_NS, 'i4')}").text = str(len(slide_titles))

    titles_parts = root.find(f"./{qname(APP_NS, 'TitlesOfParts')}/{qname(VT_NS, 'vector')}")
    if titles_parts is not None:
        base_items = list(titles_parts)
        preserved_prefix = base_items[:11]
        for child in list(titles_parts):
            titles_parts.remove(child)
        for child in preserved_prefix:
            titles_parts.append(copy.deepcopy(child))
        for title in slide_titles:
            lpstr = ET.Element(qname(VT_NS, "lpstr"))
            lpstr.text = title
            titles_parts.append(lpstr)
        titles_parts.set("size", str(len(preserved_prefix) + len(slide_titles)))

    write_xml(app_path, tree)


def build_presentation(template: Path, output: Path) -> None:
    slides = slide_deck_definition()

    with tempfile.TemporaryDirectory(prefix="ppt-build-") as tmpdir:
        tmp_root = Path(tmpdir)
        with zipfile.ZipFile(template, "r") as zf:
            zf.extractall(tmp_root)

        slides_dir = tmp_root / "ppt" / "slides"
        rels_dir = slides_dir / "_rels"

        populate_title_slide(slides_dir / "slide1.xml", slides[0]["title"], slides[0]["subtitle"])
        populate_content_slide(slides_dir / "slide2.xml", slides[1]["title"], slides[1]["bullets"])

        next_slide_number = 3
        next_slide_id = get_next_slide_id(tmp_root / "ppt" / "presentation.xml")
        next_rel_id = get_next_rel_id(tmp_root / "ppt" / "_rels" / "presentation.xml.rels")

        for slide in slides[2:]:
            slide_xml = slides_dir / f"slide{next_slide_number}.xml"
            slide_rels = rels_dir / f"slide{next_slide_number}.xml.rels"
            shutil.copyfile(slides_dir / "slide2.xml", slide_xml)
            shutil.copyfile(rels_dir / "slide2.xml.rels", slide_rels)
            populate_content_slide(slide_xml, slide["title"], slide["bullets"])

            add_slide_relationship(
                tmp_root / "ppt" / "_rels" / "presentation.xml.rels",
                f"rId{next_rel_id}",
                f"slides/slide{next_slide_number}.xml",
            )
            add_slide_reference(
                tmp_root / "ppt" / "presentation.xml",
                next_slide_id,
                f"rId{next_rel_id}",
            )
            add_content_type_override(tmp_root / "[Content_Types].xml", next_slide_number)

            next_slide_number += 1
            next_slide_id += 1
            next_rel_id += 1

        update_app_properties(
            tmp_root / "docProps" / "app.xml",
            [slide["title"] for slide in slides],
        )

        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(tmp_root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(tmp_root))


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    template = repo_root / "wm.pptx"
    output = repo_root / "SIDE-java-to-SIDE-py-update.pptx"
    build_presentation(template, output)
    print(output)


if __name__ == "__main__":
    main()
