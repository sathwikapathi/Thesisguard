import os
import re
import json
import sqlite3
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, g

# ── LangChain Imports ──────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Report Generation ──────────────────────────────────────────────────────
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

DATABASE = 'thesisguard.db'

# ═══════════════════════════════════════════════════════════════════════════
# LangChain Agent Setup
# ═══════════════════════════════════════════════════════════════════════════

def get_llm():
    """Initialize the LangChain LLM with Google Gemini"""
    api_key = os.environ.get('GEMINI_API_KEY', '')
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=1500
    )

# ── PROMPT TEMPLATES (Prompt Engineering) ─────────────────────────────────

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["thesis_text"],
    template="""You are ThesisGuard, an expert academic thesis review agent.

Your task is to read the following thesis and write a clear, concise summary.

Write 150-200 words covering:
- What the thesis is about
- The main research question or objective  
- The methodology used
- Key findings or conclusions

Be plain and easy to understand. Write for the student author.

THESIS:
{thesis_text}

SUMMARY (write only the summary, nothing else):"""
)

STRUCTURE_PROMPT = PromptTemplate(
    input_variables=["thesis_text"],
    template="""You are an expert academic writing evaluator specializing in thesis structure.

Analyze this thesis for structural completeness and logical flow.

Check for these standard thesis sections:
- Abstract
- Introduction  
- Literature Review / Background
- Methodology / Methods
- Results / Findings
- Discussion
- Conclusion
- References / Bibliography

Instructions:
- List which sections ARE present
- List which sections are MISSING
- Give a STRUCTURE SCORE from 0-100
- List up to 4 specific structural issues
- For each issue, explain WHY it matters academically
- For each issue, provide a specific Suggested Fix

Respond ONLY with valid JSON in this exact format:
{{
  "score": 75,
  "sections_found": ["Introduction", "Methodology", "Results", "Conclusion"],
  "sections_missing": ["Abstract", "Literature Review"],
  "issues": [
    {{
      "issue": "Missing Abstract section",
      "why": "An abstract is required by academic standards to give readers a quick overview before reading",
      "suggested_fix": "Add an Abstract of 150-250 words before the Introduction. Include: research question, methodology, key findings, and conclusion."
    }}
  ]
}}

THESIS:
{thesis_text}

JSON RESPONSE:"""
)

CITATION_PROMPT = PromptTemplate(
    input_variables=["thesis_text"],
    template="""You are an expert academic citation analyst.

Analyze this thesis for citation consistency, completeness, and formatting.

Check for:
- Are in-text citations present? What style are they using (APA, MLA, IEEE, Chicago)?
- Are there factual statements or statistics without citations?
- Are citation formats consistent throughout the document?
- Do in-text citations match the reference list?

Give a CITATION SCORE from 0-100.
List up to 4 specific citation issues with explanations and fixes.

Respond ONLY with valid JSON in this exact format:
{{
  "score": 70,
  "citation_style_detected": "APA",
  "issues": [
    {{
      "issue": "Statistics cited without source",
      "original": "Studies show that 70% of students struggle with thesis writing",
      "why": "Uncited statistics are considered academic dishonesty and reduce the credibility of your argument. Reviewers will question the validity of unsupported claims.",
      "suggested_fix": "Find the original source for this statistic and add a citation: (Author, Year). Then add the full reference to your bibliography."
    }}
  ]
}}

THESIS:
{thesis_text}

JSON RESPONSE:"""
)

GRAMMAR_PROMPT = PromptTemplate(
    input_variables=["thesis_text"],
    template="""You are an expert English grammar checker for academic writing.

Analyze this thesis sample for grammar and language errors.

Look specifically for:
- Subject-verb agreement errors
- Tense inconsistency (mixing past and present tense)
- Run-on sentences or sentence fragments
- Incorrect punctuation
- Wrong word usage (affect/effect, their/there/they're, etc.)
- Passive voice overuse

Give a GRAMMAR SCORE from 0-100.
List up to 5 specific grammar issues with original text and corrections.

Respond ONLY with valid JSON in this exact format:
{{
  "score": 80,
  "issues": [
    {{
      "issue": "Tense inconsistency in methodology section",
      "original": "The study was conducted and the results are showing positive outcomes.",
      "why": "Mixing past tense (was conducted) with present tense (are showing) is grammatically incorrect and confuses readers about when events occurred. Academic writing requires consistent tense.",
      "suggested_fix": "The study was conducted and the results showed positive outcomes."
    }}
  ]
}}

THESIS (first section):
{thesis_text}

JSON RESPONSE:"""
)

STYLE_PROMPT = PromptTemplate(
    input_variables=["thesis_text"],
    template="""You are an expert academic writing style consultant.

Analyze this thesis for writing style, tone, and readability issues.

Check for:
- Informal or conversational language (should be formal academic tone)
- Vague or imprecise language ("some", "many", "basically", "a lot")
- Overly complex sentences that reduce clarity
- Missing topic sentences in paragraphs
- Weak transitions between sections
- First-person overuse where third-person is more appropriate

Give a STYLE SCORE from 0-100.
List up to 4 specific style issues with examples and improvements.

Respond ONLY with valid JSON in this exact format:
{{
  "score": 75,
  "issues": [
    {{
      "issue": "Informal language in methodology section",
      "original": "We basically did a survey with some students to get their opinions.",
      "why": "Academic writing requires precise, formal language. Words like 'basically' and 'some' are vague and unprofessional. Reviewers expect exact numbers and formal phrasing in methodology sections.",
      "suggested_fix": "A structured survey questionnaire was administered to 45 undergraduate students enrolled in the Computer Science department to gather quantitative data on learning preferences."
    }}
  ]
}}

THESIS:
{thesis_text}

JSON RESPONSE:"""
)

# ═══════════════════════════════════════════════════════════════════════════
# LangChain Agent - Thesis Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class ThesisGuardAgent:
    """
    LangChain-powered thesis review agent.
    Uses 5 specialized LLMChains, each with engineered prompts,
    to analyze different aspects of the thesis document.
    """

    def __init__(self):
        self.llm = get_llm()

        # Create individual LangChain chains for each analysis task
        self.summary_chain   = LLMChain(llm=self.llm, prompt=SUMMARY_PROMPT,   verbose=True)
        self.structure_chain = LLMChain(llm=self.llm, prompt=STRUCTURE_PROMPT, verbose=True)
        self.citation_chain  = LLMChain(llm=self.llm, prompt=CITATION_PROMPT,  verbose=True)
        self.grammar_chain   = LLMChain(llm=self.llm, prompt=GRAMMAR_PROMPT,   verbose=True)
        self.style_chain     = LLMChain(llm=self.llm, prompt=STYLE_PROMPT,     verbose=True)

        # Text splitter for handling long documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=200
        )

    def analyze(self, text):
        """Run all 5 analysis chains on the thesis text"""
        # Truncate for API limits while keeping key content
        text_full    = text[:8000]
        text_grammar = text[:4000]

        results = {}

        # ── Chain 1: Summary ──────────────────────────────────────────────
        try:
            raw = self.summary_chain.run(thesis_text=text_full)
            results['summary'] = raw.strip()
        except Exception as e:
            results['summary'] = f"Could not generate summary: {str(e)}"

        # ── Chain 2: Structural Analysis ──────────────────────────────────
        try:
            raw = self.structure_chain.run(thesis_text=text_full)
            results['structure'] = self._parse_json(raw, {
                "score": 50,
                "sections_found": [],
                "sections_missing": ["Could not detect sections"],
                "issues": [{"issue": "Structure analysis failed", "why": "Document may not have clear section headings", "suggested_fix": "Add clear section headings like Introduction, Methodology, Results, Conclusion."}]
            })
        except Exception as e:
            results['structure'] = {"score": 50, "sections_found": [], "sections_missing": [], "issues": [{"issue": str(e), "why": "API error", "suggested_fix": "Please retry."}]}

        # ── Chain 3: Citation Analysis ────────────────────────────────────
        try:
            raw = self.citation_chain.run(thesis_text=text_full)
            results['citations'] = self._parse_json(raw, {
                "score": 55,
                "citation_style_detected": "Unknown",
                "issues": [{"issue": "Citation analysis failed", "why": "Could not parse citations", "suggested_fix": "Ensure citations follow APA or IEEE format consistently."}]
            })
        except Exception as e:
            results['citations'] = {"score": 55, "citation_style_detected": "Unknown", "issues": []}

        # ── Chain 4: Grammar Analysis ─────────────────────────────────────
        try:
            raw = self.grammar_chain.run(thesis_text=text_grammar)
            results['grammar'] = self._parse_json(raw, {
                "score": 60,
                "issues": [{"issue": "Grammar analysis failed", "why": "Could not parse document", "suggested_fix": "Proofread each section carefully for tense consistency and subject-verb agreement."}]
            })
        except Exception as e:
            results['grammar'] = {"score": 60, "issues": []}

        # ── Chain 5: Style Analysis ───────────────────────────────────────
        try:
            raw = self.style_chain.run(thesis_text=text_grammar)
            results['style'] = self._parse_json(raw, {
                "score": 60,
                "issues": [{"issue": "Style analysis failed", "why": "Could not parse document", "suggested_fix": "Ensure consistent formal academic tone throughout your thesis."}]
            })
        except Exception as e:
            results['style'] = {"score": 60, "issues": []}

        return results

    def _parse_json(self, raw_text, fallback):
        """Safely extract and parse JSON from LLM response"""
        try:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(raw_text)
        except:
            return fallback


# ═══════════════════════════════════════════════════════════════════════════
# Document Parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_pdf(file_bytes):
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    except Exception as e:
        return f"PDF parsing error: {e}"

def parse_docx(file_bytes):
    try:
        import io
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except Exception as e:
        return f"DOCX parsing error: {e}"

def extract_text(file_bytes, filename):
    f = filename.lower()
    if f.endswith('.pdf'):   return parse_pdf(file_bytes)
    if f.endswith('.docx'):  return parse_docx(file_bytes)
    if f.endswith('.txt'):   return file_bytes.decode('utf-8', errors='ignore')
    return "Unsupported file format."


# ═══════════════════════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════════════════════

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(e):
    db = getattr(g, '_database', None)
    if db: db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT,
            created_at      TEXT,
            summary         TEXT,
            structure_score INTEGER,
            citation_score  INTEGER,
            grammar_score   INTEGER,
            style_score     INTEGER,
            full_report     TEXT
        )''')
        db.commit()


# ═══════════════════════════════════════════════════════════════════════════
# PDF Report Generator
# ═══════════════════════════════════════════════════════════════════════════

def score_label(s):
    return "Good" if s >= 80 else "Needs Work" if s >= 60 else "Critical"

def generate_pdf_report(filename, results, path):
    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story  = []

    def style(name, **kw):
        return ParagraphStyle(name, parent=styles['Normal'], **kw)

    title_s   = style('T', fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#1a365d'), alignment=TA_CENTER, spaceAfter=6)
    sub_s     = style('S', fontSize=12, textColor=colors.HexColor('#4a5568'), alignment=TA_CENTER, spaceAfter=4)
    heading_s = style('H', fontName='Helvetica-Bold', fontSize=13, textColor=colors.HexColor('#2b6cb0'), spaceBefore=14, spaceAfter=6)
    body_s    = style('B', fontSize=10, leading=16, spaceAfter=8)
    issue_s   = style('I', fontSize=9,  leading=14, leftIndent=16, textColor=colors.HexColor('#4a5568'))
    fix_s     = style('F', fontSize=9,  leading=14, leftIndent=16, textColor=colors.HexColor('#276749'))

    # Cover
    story += [
        Spacer(1, 0.4*inch),
        Paragraph("ThesisGuard AI", title_s),
        Paragraph("Thesis Analysis Report", sub_s),
        Spacer(1, 0.2*inch),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2b6cb0')),
        Spacer(1, 0.15*inch),
        Paragraph(f"Document: {filename}", sub_s),
        Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", sub_s),
        Paragraph("Powered by LangChain + Google Gemini AI", style('P', fontSize=9, textColor=colors.HexColor('#718096'), alignment=TA_CENTER)),
        Spacer(1, 0.4*inch),
    ]

    # Score table
    overall = (results['structure'].get('score',0) + results['citations'].get('score',0) +
               results['grammar'].get('score',0)   + results['style'].get('score',0)) // 4

    rows = [
        ['Category', 'Score', 'Status'],
        ['Structure',  f"{results['structure'].get('score',0)}/100",  score_label(results['structure'].get('score',0))],
        ['Citations',  f"{results['citations'].get('score',0)}/100",  score_label(results['citations'].get('score',0))],
        ['Grammar',    f"{results['grammar'].get('score',0)}/100",    score_label(results['grammar'].get('score',0))],
        ['Style',      f"{results['style'].get('score',0)}/100",      score_label(results['style'].get('score',0))],
        ['OVERALL',    f"{overall}/100",                              score_label(overall)],
    ]
    t = Table(rows, colWidths=[2.5*inch, 1.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR',(0,0),(-1,0),  colors.white),
        ('FONTNAME',(0,0),(-1,0),   'Helvetica-Bold'),
        ('ALIGN',(0,0),(-1,-1),     'CENTER'),
        ('FONTSIZE',(0,0),(-1,-1),  10),
        ('ROWBACKGROUNDS',(0,1),(-1,-2),[colors.HexColor('#ebf8ff'), colors.white]),
        ('BACKGROUND',(0,-1),(-1,-1),colors.HexColor('#dbeafe')),
        ('FONTNAME',(0,-1),(-1,-1), 'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),0.5,  colors.HexColor('#bee3f8')),
        ('TOPPADDING',(0,0),(-1,-1),7),
        ('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    story += [t, Spacer(1, 0.4*inch)]

    # Summary
    story += [
        Paragraph("1. Thesis Summary", heading_s),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bee3f8')),
        Spacer(1, 0.1*inch),
        Paragraph(results.get('summary','N/A'), body_s),
    ]

    # Sections helper
    def add_section(title, data, show_sections=False):
        story.append(Paragraph(title, heading_s))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bee3f8')))
        story.append(Spacer(1, 0.1*inch))
        if show_sections:
            found   = data.get('sections_found', [])
            missing = data.get('sections_missing', [])
            if found:
                story.append(Paragraph(f"<b>Sections Found:</b> {', '.join(found)}", body_s))
            if missing:
                story.append(Paragraph(f"<b>Missing Sections:</b> {', '.join(missing)}", style('M', fontSize=10, textColor=colors.HexColor('#c53030'), spaceAfter=8)))
        for item in data.get('issues', []):
            story.append(Paragraph(f"<b>Issue:</b> {item.get('issue','')}", body_s))
            if item.get('original'):
                story.append(Paragraph(f'Original: "{item["original"]}"', issue_s))
            story.append(Paragraph(f"Why this matters: {item.get('why','')}", issue_s))
            story.append(Paragraph(f"Suggested Fix: {item.get('suggested_fix','')}", fix_s))
            story.append(Spacer(1, 0.1*inch))

    add_section("2. Structural Flow Analysis",    results.get('structure',{}), show_sections=True)
    add_section("3. Citation Consistency",         results.get('citations',{}))
    add_section("4. Grammar & Language Analysis",  results.get('grammar',{}))
    add_section("5. Writing Style Analysis",        results.get('style',{}))

    story += [
        Spacer(1, 0.4*inch),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2b6cb0')),
        Paragraph("ThesisGuard AI — Powered by LangChain + Google Gemini | Ethical AI for Academic Excellence",
                  style('Footer', fontSize=8, textColor=colors.HexColor('#718096'), alignment=TA_CENTER))
    ]
    doc.build(story)


# ═══════════════════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_THESIS = """
Abstract
This study investigates the impact of artificial intelligence on student learning outcomes in undergraduate engineering programs. The research aims to understand how AI-powered tutoring systems affect academic performance and critical thinking skills among engineering students.

Introduction
Artificial intelligence has revolutionized many sectors of modern society. In education, AI tools are increasingly being adopted to personalize learning experiences for students. This thesis examines whether AI tutoring systems improve outcomes for engineering students across three universities. Studies show that 70% of students who used AI tutoring improved their grades. The research question is: Does AI-based tutoring significantly improve academic performance in undergraduate engineering education?

Literature Review
Previous research has extensively explored the role of technology in education. Johnson (2019) found that personalized learning systems improved test scores by 15% among university students. Smith and Kumar (2020) demonstrated that adaptive AI systems can effectively adjust to individual learning paces, resulting in better retention of complex concepts. However, Chen et al. (2021) cautioned that over-reliance on AI tools may reduce critical thinking development in students. The literature presents a mixed picture of benefits and potential drawbacks.

Methodology
We basically did a survey with some students at three universities. We also collected grade data from their courses. The study was conducted over one full semester and the results are showing positive outcomes across most participants. Students were asked to use an AI tutoring system for four hours per week. Data was collected before and after the intervention period using standardized assessment tools.

Results
The results showed improvement in grades across all three institutions. Students who used the AI tutoring system performed better on final examinations. Average grades improved from 68% to 74% after the intervention. Student satisfaction scores were also higher in the experimental group. However, some students reported feeling overly dependent on the AI system for problem solving.

Conclusion
AI tutoring systems appear to have a positive impact on academic performance in engineering education. Universities should consider adopting these systems more widely. Future research should examine the long-term effects of AI tutoring on critical thinking and independent problem-solving skills.

References
Johnson, M. (2019). Personalized Learning in Higher Education. Journal of Education Technology, 12(3), 45-67.
Smith, R., & Kumar, A. (2020). Adaptive AI Systems in Classroom Settings. Educational Research Quarterly, 8(2), 112-128.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    file_bytes = file.read()
    text = extract_text(file_bytes, file.filename)

    if len(text.strip()) < 100:
        return jsonify({'error': 'Document is empty or could not be parsed. Please upload a text-based PDF or DOCX.'}), 400

    try:
        agent = ThesisGuardAgent()
        results = agent.analyze(text)
    except Exception as e:
        return jsonify({'error': f'AI analysis failed: {str(e)}'}), 500

    db = get_db()
    db.execute(
        'INSERT INTO analyses (filename,created_at,summary,structure_score,citation_score,grammar_score,style_score,full_report) VALUES (?,?,?,?,?,?,?,?)',
        (file.filename, datetime.now().isoformat(),
         results.get('summary',''),
         results['structure'].get('score',0),
         results['citations'].get('score',0),
         results['grammar'].get('score',0),
         results['style'].get('score',0),
         json.dumps(results))
    )
    aid = db.execute('SELECT last_insert_rowid()').fetchone()[0]
    db.commit()

    return jsonify({'success': True, 'analysis_id': aid, 'filename': file.filename, 'results': results})

@app.route('/demo', methods=['POST'])
def demo():
    try:
        agent = ThesisGuardAgent()
        results = agent.analyze(SAMPLE_THESIS)
    except Exception as e:
        return jsonify({'error': f'Demo failed: {str(e)}'}), 500
    return jsonify({'success': True, 'analysis_id': 0, 'filename': 'sample_thesis_demo.txt', 'results': results})

@app.route('/download/<int:aid>')
def download(aid):
    if aid == 0:
        return jsonify({'error': 'Upload a real thesis to download a report.'}), 400
    row = get_db().execute('SELECT * FROM analyses WHERE id=?', (aid,)).fetchone()
    if not row:
        return jsonify({'error': 'Analysis not found'}), 404
    results = json.loads(row['full_report'])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    generate_pdf_report(row['filename'], results, tmp.name)
    tmp.close()
    return send_file(tmp.name, as_attachment=True,
                     download_name=f"ThesisGuard_Report_{row['filename']}.pdf",
                     mimetype='application/pdf')

if __name__ == '__main__':
    init_db()
    print("ThesisGuard AI is running on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
