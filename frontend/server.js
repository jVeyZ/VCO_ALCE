const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs/promises');
const QRCode = require('qrcode');
const puppeteer = require('puppeteer');
const crypto = require('crypto');
const { execFile } = require('child_process');

const app = express();
const PORT = process.env.PORT || 5500;
const __dirnameResolved = __dirname;
const templatePath = path.join(__dirnameResolved, 'templates', 'answer-sheet.html');

// Labeling paths
const REPO_ROOT = path.join(__dirnameResolved, '..');
const LABEL_QUEUE_DIR = path.join(REPO_ROOT, 'img', 'label_queue');
const LABEL_ARCHIVE_DIR = path.join(LABEL_QUEUE_DIR, '_labeled');
const LABEL_OUTPUT_DIR = path.join(REPO_ROOT, 'data', 'manual_digits');
const LABEL_IMAGE_DIR = path.join(LABEL_OUTPUT_DIR, 'images');
const LABEL_LOG_PATH = path.join(LABEL_OUTPUT_DIR, 'labels.jsonl');
const SUPPORTED_IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.bmp']);
const ALLOWED_DIGIT_LABELS = new Set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
const SEGMENT_SCRIPT = path.join(REPO_ROOT, 'src', 'segment_digits.py');

app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.static(__dirnameResolved));
app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirnameResolved, 'index.html'));
});

app.get('/labeler', (_req, res) => {
  res.sendFile(path.join(__dirnameResolved, 'labeler.html'));
});

async function ensureLabelingDirs() {
  await fs.mkdir(LABEL_QUEUE_DIR, { recursive: true });
  await fs.mkdir(LABEL_ARCHIVE_DIR, { recursive: true });
  await fs.mkdir(LABEL_IMAGE_DIR, { recursive: true });
}

async function readLabelLog() {
  try {
    const content = await fs.readFile(LABEL_LOG_PATH, 'utf8');
    return content
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  } catch (err) {
    if (err.code === 'ENOENT') return [];
    throw err;
  }
}

async function appendLabelLog(entry) {
  const line = `${JSON.stringify(entry)}\n`;
  await fs.mkdir(LABEL_OUTPUT_DIR, { recursive: true });
  await fs.appendFile(LABEL_LOG_PATH, line, 'utf8');
}

async function listQueueFiles() {
  await ensureLabelingDirs();
  const files = await fs.readdir(LABEL_QUEUE_DIR, { withFileTypes: true });
  return files
    .filter((f) => f.isFile())
    .map((f) => f.name)
    .filter((name) => SUPPORTED_IMAGE_EXTS.has(path.extname(name).toLowerCase()));
}

function runSegmentation(imagePath, mode = 'full') {
  return new Promise((resolve, reject) => {
    execFile('python', [SEGMENT_SCRIPT, imagePath, mode], { timeout: 15_000 }, (error, stdout, stderr) => {
      if (error) {
        console.error('Segmentation script failed:', error, stderr?.toString());
        return reject(error);
      }
      try {
        const parsed = JSON.parse(stdout.toString());
        resolve(parsed);
      } catch (parseErr) {
        console.error('Failed to parse segmentation output:', parseErr);
        reject(parseErr);
      }
    });
  });
}

async function loadNextImage() {
  const labeledEntries = await readLabelLog();
  const labeledMap = new Map(); // key: source -> set of segment indices
  labeledEntries.forEach((e) => {
    const key = e.source;
    if (!labeledMap.has(key)) labeledMap.set(key, new Set());
    if (typeof e.segmentIndex === 'number') {
      labeledMap.get(key).add(e.segmentIndex);
    }
  });

  const queueFiles = await listQueueFiles();

  for (const filename of queueFiles) {
    const filePath = path.join(LABEL_QUEUE_DIR, filename);
    const segmentation = await runSegmentation(filePath, 'full').catch(() => null);
    if (!segmentation || !segmentation.segments) continue;

    const already = labeledMap.get(filename) || new Set();
    const pendingSegments = segmentation.segments.filter((s) => !already.has(s.index));

    if (!pendingSegments.length) {
      // Archive and continue to next file
      await fs.rename(filePath, path.join(LABEL_ARCHIVE_DIR, filename)).catch(async () => {
        await fs.unlink(filePath).catch(() => {});
      });
      continue;
    }

    const buffer = await fs.readFile(filePath);
    const ext = path.extname(filename).toLowerCase().replace('.', '') || 'png';
    const dataUrl = `data:image/${ext};base64,${buffer.toString('base64')}`;

    return {
      filename,
      dataUrl,
      remaining: queueFiles.length,
      total: queueFiles.length,
      segmentation: { ...segmentation, segments: pendingSegments },
    };
  }

  return null;
}

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

app.get('/api/label/next', async (_req, res) => {
  try {
    const next = await loadNextImage();
    if (!next) {
      return res.status(204).end();
    }
    return res.json(next);
  } catch (error) {
    console.error('Failed to load next image:', error);
    return res.status(500).json({ message: 'Could not load next image.' });
  }
});

app.post('/api/label/segment', async (req, res) => {
  const { filename, label, segmentIndex, kind, imageData } = req.body || {};
  if (!filename || typeof filename !== 'string') {
    return res.status(400).json({ message: 'Missing filename.' });
  }
  if (!ALLOWED_DIGIT_LABELS.has(String(label))) {
    return res.status(400).json({ message: 'Label must be a digit 0-9.' });
  }
  if (typeof segmentIndex !== 'number') {
    return res.status(400).json({ message: 'Missing segmentIndex.' });
  }
  if (typeof imageData !== 'string' || !imageData.startsWith('data:image/')) {
    return res.status(400).json({ message: 'Missing segment image data URL.' });
  }

  try {
    await ensureLabelingDirs();

    const sourcePath = path.join(LABEL_QUEUE_DIR, filename);
    const sourceExists = await fs.stat(sourcePath).then(() => true).catch(() => false);
    if (!sourceExists) {
      return res.status(404).json({ message: 'Source image not found in queue.' });
    }

    const [, meta, b64] = imageData.match(/^data:(image\/[^;]+);base64,(.+)$/) || [];
    if (!b64) {
      return res.status(400).json({ message: 'Invalid image data.' });
    }
    const safeName = `${Date.now()}-${crypto.randomBytes(4).toString('hex')}-seg${segmentIndex}-${sanitizeFileName(filename)}.png`;
    const labelDir = path.join(LABEL_IMAGE_DIR, String(label));
    await fs.mkdir(labelDir, { recursive: true });
    const targetPath = path.join(labelDir, safeName);
    await fs.writeFile(targetPath, Buffer.from(b64, 'base64'));

    const entry = {
      label: String(label),
      source: filename,
      segmentIndex,
      kind: kind || 'segment',
      savedAs: path.relative(REPO_ROOT, targetPath),
      timestamp: new Date().toISOString(),
    };
    await appendLabelLog(entry);

    // If all segments are labeled, archive the source image
    const labeledEntries = await readLabelLog();
    const labeledForFile = new Set(
      labeledEntries.filter((e) => e.source === filename).map((e) => e.segmentIndex)
    );
    const segmentation = await runSegmentation(sourcePath, 'full').catch(() => null);
    const pending = segmentation?.segments?.filter((s) => !labeledForFile.has(s.index)) || [];
    if (!pending.length) {
      await fs.rename(sourcePath, path.join(LABEL_ARCHIVE_DIR, filename)).catch(async () => {
        await fs.unlink(sourcePath).catch(() => {});
      });
    }

    const next = await loadNextImage();
    return res.json({ message: 'Saved', next });
  } catch (error) {
    console.error('Failed to save label:', error);
    return res.status(500).json({ message: 'Could not save label.' });
  }
});

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
