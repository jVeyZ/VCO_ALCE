const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs/promises');
const QRCode = require('qrcode');
const puppeteer = require('puppeteer');

const app = express();
const PORT = process.env.PORT || 5500;
const __dirnameResolved = __dirname;
const templatePath = path.join(__dirnameResolved, 'templates', 'answer-sheet.html');

app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.static(__dirnameResolved));
app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirnameResolved, 'index.html'));
});

function shuffle(array) {
  const copy = [...array];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function htmlEscape(value = '') {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function selectQuestions(strategy, questionAmount, questionBank) {
  if (!Array.isArray(questionBank) || !questionBank.length) return [];
  const total = Math.min(questionAmount || questionBank.length, questionBank.length);
  const baseSet = questionBank.slice(0, total);
  if (strategy === 'shuffle') {
    return shuffle(baseSet);
  }
  return shuffle(questionBank).slice(0, total);
}

function renderQuestionRows(questions) {
  const leftColumn = [];
  const rightColumn = [];

  questions.forEach((question, index) => {
    const questionHtml = `
      <tr>
        <td><b style="text-align:center;">${index + 1}</b></td>
        <td>
          ${question.type === 'mcq'
        ? `<div class="mcq-options">
                <div class="bubble">A</div>
                <div class="bubble">B</div>
                <div class="bubble">C</div>
                <div class="bubble">D</div>
              </div>`
        : `<div class="numeric-slots">
                <div class="slot"></div>
                <div class="slot"></div>
                <div class="slot"></div>
                <div class="slot"></div>
                <div class="slot"></div>
                <div class="slot"></div>
                <div class="slot"></div>
                <div class="slot"></div>
                <div class="slot"></div>
              </div>`
      }
        </td>
      </tr>`;

    if (index < 13) {
      leftColumn.push(questionHtml);
    } else {
      rightColumn.push(questionHtml);
    }
  });

  return {
    left: leftColumn.join('\n'),
    right: rightColumn.join('\n'),
    hideRight: questions.length <= 13 ? 'hidden' : ''
  };
}

async function getInstructionsImage() {
  try {
    const instructionsPath = path.join(__dirnameResolved, 'templates', 'instructions.png');
    const imageBuffer = await fs.readFile(instructionsPath);
    const base64 = imageBuffer.toString('base64');
    return `<img src="data:image/png;base64,${base64}" alt="Instructions" style="width:300px;" />`;
  } catch (error) {
    console.warn('Instructions image not found, skipping.');
    return '';
  }
}

async function buildTemplateHtml(blueprint, variantLabel) {
  const template = await fs.readFile(templatePath, 'utf8');
  const variant = blueprint?.exam?.variants?.find((v) => v.label === variantLabel) || blueprint?.exam?.variants?.[0];
  if (!variant) {
    throw new Error('Unable to find the requested exam variant.');
  }
  const selectedQuestions = selectQuestions(
    variant.strategy,
    blueprint.exam.questionAmountPerExam,
    blueprint.questionBank
  );
  const numericIndices = selectedQuestions
    .map((q, idx) => q.type === 'numeric' ? idx + 1 : null)
    .filter(idx => idx !== null);
  const numericCount = numericIndices.length;
  const totalCount = selectedQuestions.length;
  const dateCreated = blueprint?.metadata?.generatedAt || new Date().toISOString();
  const dateFormatted = new Date(dateCreated).toISOString().split('T')[0].replace(/-/g, '');
  const qrPayload = [
    '1.0',
    blueprint?.subject?.code || 'N/A',
    blueprint?.exam?.school || 'N/A',
    dateFormatted,
    variant.label,
    numericCount,
    totalCount,
    numericIndices.join(',')
  ].join(';');
  const qrCode = await QRCode.toDataURL(qrPayload, { margin: 1, width: 256 });
  const instructionsImage = await getInstructionsImage();
  const questionRowsData = renderQuestionRows(selectedQuestions);
  const replacements = {
    '{{EXAM_NAME}}': htmlEscape(blueprint?.exam?.name || 'Exam'),
    '{{SUBJECT_NAME}}': htmlEscape(blueprint?.subject?.name || 'Subject'),
    '{{SUBJECT_CODE}}': htmlEscape(blueprint?.subject?.code || 'CODE'),
    '{{SCHOOL_NAME}}': htmlEscape(blueprint?.exam?.school || 'School'),
    '{{EXAM_VARIANT}}': htmlEscape(variant.label),
    '{{DATE_CREATED}}': htmlEscape(new Date(dateCreated).toLocaleDateString()),
    '{{QUESTION_TOTAL}}': String(totalCount),
    '{{QR_PAYLOAD}}': htmlEscape(qrPayload),
    '{{QR_CODE}}': qrCode,
    '{{INSTRUCTIONS_IMAGE}}': instructionsImage,
    '{{QUESTION_ROWS_LEFT}}': questionRowsData.left,
    '{{QUESTION_ROWS_RIGHT}}': questionRowsData.right,
    '{{RIGHT_COLUMN_CLASS}}': questionRowsData.hideRight ? 'hidden' : ''
  };
  let filled = template;
  Object.entries(replacements).forEach(([token, value]) => {
    filled = filled.split(token).join(value);
  });
  return filled;
}

async function renderPdf(html) {
  const browser = await puppeteer.launch({ headless: 'new' });
  try {
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: 'networkidle0' });
    return await page.pdf({
      format: 'A4',
      printBackground: true,
      margin: { top: '2mm', right: '2mm', bottom: '2mm', left: '2mm' }
    });
  } finally {
    await browser.close();
  }
}

function sanitizeFileName(value = 'exam') {
  return value.replace(/[^a-z0-9-_]+/gi, '-').replace(/-{2,}/g, '-').replace(/^-|-$/g, '').toLowerCase();
}

app.post('/api/generate-answer-sheet', async (req, res) => {
  const { blueprint, variantLabel } = req.body || {};
  if (!blueprint || !variantLabel) {
    return res.status(400).json({ message: 'Provide "blueprint" data and the desired "variantLabel".' });
  }
  try {
    const html = await buildTemplateHtml(blueprint, variantLabel);
    const pdfBuffer = await renderPdf(html);
    const fileName = `${sanitizeFileName(blueprint.subject?.code || 'exam')}-${sanitizeFileName(variantLabel)}.pdf`;
    
    // Save answer key JSON to answer_key folder using exam ID as filename
    const answerKeyDir = path.join(__dirnameResolved, 'answer_key');
    try {
      await fs.mkdir(answerKeyDir, { recursive: true });
    } catch (e) {
      console.warn('Could not create answer_key directory:', e);
    }
    
    const examId = blueprint?.subject?.code || 'exam';
    const jsonFileName = `${sanitizeFileName(examId)}.json`;
    const jsonFilePath = path.join(answerKeyDir, jsonFileName);
    await fs.writeFile(jsonFilePath, JSON.stringify(blueprint, null, 2), 'utf8');
    console.log(`Answer key saved: ${jsonFilePath}`);
    
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);
    return res.send(pdfBuffer);
  } catch (error) {
    console.error('PDF generation failed:', error);
    return res.status(500).json({ message: 'Failed to generate the answer sheet PDF.' });
  }
});

app.listen(PORT, () => {
  console.log(`Express server ready on http://localhost:${PORT}`);
});
