            present_in_text = any(term in v["text"].lower() for v in documents.values())
            # Only phrase as a check because a text mention doesn't establish that a file is absent.
            if present_in_text:
                findings.append({
                    "level": "info", "title": f"Check supporting evidence: {term.title()}",
                    "detail": f"The phrase “{term}” appears in the uploaded text. Confirm whether the actual document is included and complete; text mention alone cannot prove that it is missing.",
                    "sources": [n for n, v in documents.items() if term in v["text"].lower()]
                })
    if not findings:
        findings.append({
            "level": "good", "title": "No obvious date/amount differences detected",
            "detail": "The simple checks did not find a clear difference in dates or currency amounts. This is not proof that the documents agree or that the claim is covered.",
            "sources": list(documents.keys())
        })
    return findings[:15]

def report_payload(documents, findings):
    return {
        "product": "CLAIMX Claim Detective",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notice": "Prototype findings are review prompts, not legal or coverage decisions.",
        "documents": [
            {"filename": name, "method": info["method"], "characters_extracted": len(info["text"])}
            for name, info in documents.items()
        ],
        "findings": findings,
    }

if "documents" not in st.session_state:
    st.session_state.documents = {}
if "findings" not in st.session_state:
    st.session_state.findings = []

left, right = st.columns([1.1, .9], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📂 Upload your case files")
    st.caption("Upload a claim letter, policy document, bills, medical report, or other supporting files.")
    uploads = st.file_uploader(
        "Choose documents",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, image, and TXT files. Keep each file under your hosting provider's upload limit."
    )
    if uploads:
        st.caption(f"{len(uploads)} file(s) selected.")
    analyze = st.button("🔎 Start investigation", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not uploads:
            st.warning("Upload at least one file first.")
        else:
            extracted = {}
            progress = st.progress(0, text="Preparing investigation…")
            for idx, uploaded in enumerate(uploads):
                with st.spinner(f"Reading {uploaded.name}…"):
                    text_value, method = extract_upload(uploaded)
                    extracted[uploaded.name] = {"text": text_value, "method": method}
                progress.progress((idx + 1) / len(uploads), text=f"Read {idx+1} of {len(uploads)} files")
            st.session_state.documents = extracted
            st.session_state.findings = investigate(extracted)
            progress.empty()
            st.success("Investigation complete. Review each finding against the source documents.")

    docs = st.session_state.documents
    if docs:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("🧠 Investigation pipeline")
        steps = [
            ("Document extraction", "Completed"),
            ("Date and amount scan", "Completed"),
            ("Cross-document review prompts", "Completed"),
            ("Evidence review", "Human verification required"),
        ]
        for label, status in steps:
            c1, c2 = st.columns([3, 1])
            c1.write("✅ " + label)
            c2.caption(status)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    docs = st.session_state.documents
    findings = st.session_state.findings
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📊 Case overview")
    a, b, c = st.columns(3)
    a.markdown(f'<div class="kpi"><div class="kpi-label">Documents</div><div class="kpi-value">{len(docs)}</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="kpi"><div class="kpi-label">Review prompts</div><div class="kpi-value">{len(findings)}</div></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="kpi"><div class="kpi-label">OCR engine</div><div class="kpi-value" style="font-size:1.05rem">{"Ready" if DOCTR_AVAILABLE else "Unavailable"}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not DOCTR_AVAILABLE:
        st.warning("DocTR could not be imported in this environment. Text-based PDFs and TXT files may still work; image/scanned-PDF OCR needs DocTR installed correctly.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🚩 Investigation findings")
    if not findings:
        st.info("Upload at least two documents and start an investigation to compare evidence.")
    for i, finding in enumerate(findings, start=1):
        level = finding["level"]
        css = "finding-good" if level == "good" else ("finding-warn" if level == "warn" else "finding-info")
        icon = "🟢" if level == "good" else ("🟠" if level == "warn" else "🔵")
        st.markdown(f'<div class="finding {css}"><strong>{icon} {i}. {finding["title"]}</strong><br><span class="small-muted">{finding["detail"]}</span></div>', unsafe_allow_html=True)
        if finding.get("sources"):
            st.caption("Sources: " + ", ".join(finding["sources"]))
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.documents:
    st.markdown("## 🧾 Evidence explorer")
    tabs = st.tabs(list(st.session_state.documents.keys()))
    for tab, (name, info) in zip(tabs, st.session_state.documents.items()):
        with tab:
            st.caption(f"Extraction method: {info['method']} · {len(info['text'])} characters")
            if info["text"]:
                st.text_area("Extracted text", info["text"], height=260, key="text_" + hashlib.md5(name.encode()).hexdigest())
            else:
                st.warning("No text could be extracted from this file.")
    payload = report_payload(st.session_state.documents, st.session_state.findings)
    st.download_button(
        "⬇️ Download investigation report (JSON)",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name="claimx_investigation_report.json",
        mime="application/json",
        use_container_width=True,
    )

st.markdown("---")
