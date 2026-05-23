import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import OpenAI from 'openai';
import { WebSocketServer, WebSocket } from 'ws';

const app = express();
const server = http.createServer(app);

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const PORT = Number(process.env.PORT || 8787);
const UPLOAD_DIR = path.resolve('./uploads');

await fsp.mkdir(UPLOAD_DIR, { recursive: true });

function sanitizeExtension(ext) {
  const safe = (ext || '').toLowerCase();

  if (safe === '.m4a') return '.m4a';
  if (safe === '.mp3') return '.mp3';
  if (safe === '.mp4') return '.mp4';
  if (safe === '.mpeg') return '.mpeg';
  if (safe === '.mpga') return '.mpga';
  if (safe === '.wav') return '.wav';
  if (safe === '.webm') return '.webm';
  if (safe === '.caf') return '.caf';

  return '.bin';
}

async function readFileHeadHex(filePath, byteCount = 32) {
  const handle = await fsp.open(filePath, 'r');

  try {
    const buffer = Buffer.alloc(byteCount);
    const { bytesRead } = await handle.read(buffer, 0, byteCount, 0);
    return buffer.subarray(0, bytesRead).toString('hex');
  } finally {
    await handle.close();
  }
}

function normalizeText(text) {
  if (typeof text !== 'string') {
    return '';
  }

  return text.replace(/\s+/g, ' ').trim();
}

function safeCategoryList(categories) {
  if (!Array.isArray(categories)) {
    return [];
  }

  return categories
    .filter((category) => category && typeof category.name === 'string')
    .map((category) => ({
      id: String(category.id ?? ''),
      name: String(category.name ?? ''),
      kind: String(category.kind ?? 'custom'),
    }))
    .filter((category) => category.id && category.name);
}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, UPLOAD_DIR);
  },
  filename: (_req, file, cb) => {
    const originalExt = sanitizeExtension(path.extname(file.originalname));
    const unique = `${Date.now()}-${Math.round(Math.random() * 1e9)}${originalExt}`;
    cb(null, unique);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 25 * 1024 * 1024,
  },
});

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'notina-relay',
    port: PORT,
  });
});

app.post('/api/transcribe-basic', upload.single('file'), async (req, res) => {
  const uploadedFile = req.file;

  try {
    if (!uploadedFile) {
      res.status(400).json({
        ok: false,
        error: 'Missing file',
        message: 'Nem érkezett hangfájl a kérésben.',
      });
      return;
    }

    const { language = 'hu', model = 'gpt-4o-transcribe', prompt = '' } = req.body ?? {};

    const stat = await fsp.stat(uploadedFile.path);
    const headHex = await readFileHeadHex(uploadedFile.path, 32);

    console.log('[relay-basic] POST /api/transcribe-basic file:', {
      language,
      model,
      prompt: prompt || '(none)',
      originalname: uploadedFile.originalname,
      mimetype: uploadedFile.mimetype,
      size: uploadedFile.size,
      path: uploadedFile.path,
      filename: uploadedFile.filename,
      statSize: stat.size,
      headHex,
    });

    if (stat.size <= 0) {
      res.status(400).json({
        ok: false,
        error: 'Empty file',
        message: 'A feltöltött hangfájl üres.',
      });
      return;
    }

    if (stat.size > 0 && stat.size < 8000) {
      res.status(422).json({
        ok: false,
        error: 'audio_too_short',
        audio_quality: 'very_low',
        message: 'A hangfelvétel túl rövid vagy csendes volt.',
      });
      return;
    }

    // verbose_json (with no_speech_prob segments) is only supported by whisper-1.
    // gpt-4o-transcribe and gpt-4o-mini-transcribe only accept 'json' or 'text'.
    const useVerboseJson = model === 'whisper-1';

    const transcriptionParams = {
      file: fs.createReadStream(uploadedFile.path),
      model,
      language,
      response_format: useVerboseJson ? 'verbose_json' : 'json',
    };
    if (prompt) transcriptionParams.prompt = normalizeText(prompt);

    const transcription = await client.audio.transcriptions.create(transcriptionParams);

    const segments = useVerboseJson ? (transcription.segments ?? []) : [];
    let validatedText = normalizeText(transcription.text);
    let avgNoSpeechProb = null;

    // gpt-4o-transcribe echoes the prompt (fully or partially) when no speech is detected — discard it
    if (prompt && validatedText.length > 10) {
      const normPrompt = normalizeText(prompt).toLowerCase();
      const normText = validatedText.toLowerCase();
      if (normPrompt.includes(normText) || normText === normPrompt) {
        console.log('[relay-basic] prompt echo detected, filtered out:', validatedText);
        validatedText = '';
      }
    }

    if (segments.length > 0) {
      avgNoSpeechProb = segments.reduce((sum, s) => sum + (s.no_speech_prob ?? 0), 0) / segments.length;
      if (avgNoSpeechProb > 0.5) {
        console.log('[relay-basic] noise detected via no_speech_prob, filtered out:', validatedText, `(avg=${avgNoSpeechProb.toFixed(3)})`);
        validatedText = '';
      }
    }

    console.log('[relay-basic] /api/transcribe-basic result:', {
      text: validatedText,
      filtered: validatedText !== normalizeText(transcription.text),
      avg_no_speech_prob: avgNoSpeechProb !== null ? Number(avgNoSpeechProb.toFixed(3)) : null,
    });

    res.json({
      ok: true,
      text: validatedText,
      audio_quality: 'ok',
      confidence_score: null,
      command: null,
    });
  } catch (error) {
    console.error('Failed to transcribe basic audio:', error);

    res.status(500).json({
      ok: false,
      error: 'Failed to transcribe basic audio',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  } finally {
    if (uploadedFile?.path) {
      try {
        await fsp.unlink(uploadedFile.path);
      } catch (cleanupError) {
        console.error('Failed to delete uploaded temp file:', cleanupError);
      }
    }
  }
});

app.post('/api/correct-hungarian-reminder', async (req, res) => {
  try {
    const { text = '' } = req.body ?? {};
    const inputText = normalizeText(text);

    console.log('[relay-basic] POST /api/correct-hungarian-reminder input:', {
      text: inputText,
    });

    if (!inputText) {
      res.json({
        ok: true,
        original_text: '',
        corrected_text: '',
      });
      return;
    }

    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [
            {
              type: 'input_text',
              text:
                'Magyar nyelvű rövid emlékeztető szövegeket rekonstruálsz hibás beszédfelismerési kimenetből. ' +
                'A bemenet lehet torz, részben értelmetlen, fonetikusan félrehallott vagy vegyes karaktereket tartalmazó szöveg. ' +
                'A corrected_text mezőben kizárólag értelmes magyar szavakból álló, természetes, rövid emlékeztető vagy megjegyzendő mondat legyen. ' +
                'Ne hagyj benne értelmetlen, idegen, torzult vagy STT-zagyvaságnak tűnő szót. ' +
                'Ha a bemenetben van értelmes magyar szó vagy szófoszlány, őrizd meg annak jelentését és lehetőleg mondatbeli szerepét. ' +
                'A torzult szavakat cseréld olyan magyar szavakra, amelyek hangalakban, hosszban, ritmusban és mondatbeli szerepben nagyjából megfelelhetnek az eredeti hibás STT-szónak. ' +
                'A mondat legyen rövid, hétköznapi, emlékeztető vagy megjegyzendő jellegű. ' +
                'Ne magyarázz. Ne adj alternatívákat. ' +
                'Példák: ' +
                '"Možklek je vagnet centit." -> "Most le kell vágni a centit." ' +
                '"Bo megyünk moziba." -> "Ma megyünk moziba." ' +
                '"Holap hívam visza Lacit." -> "Holnap hívjam vissza Lacit."',
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'input_text',
              text: inputText,
            },
          ],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'hungarian_reminder_correction',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              corrected_text: {
                type: 'string',
              },
            },
            required: ['corrected_text'],
          },
        },
      },
    });

    let parsed = null;

    try {
      parsed = JSON.parse(response.output_text || '{}');
    } catch (parseError) {
      console.error('Failed to parse Hungarian correction response:', parseError);
      throw new Error('A magyar javítás válasz nem volt értelmezhető JSON.');
    }

    const correctedText = normalizeText(parsed.corrected_text || inputText);

    console.log('[relay-basic] /api/correct-hungarian-reminder result:', {
      original_text: inputText,
      corrected_text: correctedText,
    });

    res.json({
      ok: true,
      original_text: inputText,
      corrected_text: correctedText,
    });
  } catch (error) {
    console.error('Failed to correct Hungarian reminder:', error);

    res.status(500).json({
      ok: false,
      error: 'Failed to correct Hungarian reminder',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.post('/api/ai-check-memo', async (req, res) => {
  try {
    const { text = '' } = req.body ?? {};
    const inputText = normalizeText(text);

    console.log('[relay-basic] POST /api/ai-check-memo input:', { text: inputText });

    if (!inputText) {
      res.json({ ok: true, corrected_text: '' });
      return;
    }

    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [
            {
              type: 'input_text',
              text:
                'Magyar emlékeztető vagy kérdés szövegét kapod STT kimenetből. ' +
                'Először döntsd el, hogy a szöveg értelmes-e: ' +
                'ha a szavak nagy része nem létező vagy zagyva (pl. "Mamóci Pametünk", "Mőzklek blurb"), ' +
                'a valid értéke legyen false és a corrected_text legyen üres string. ' +
                'Ha a szöveg értelmes magyar mondat (akár kisebb STT hibával), a valid értéke legyen true. ' +
                'Értelmes szövegnél CSAK akkor módosítsd, ha egyértelműen félrehallott szó szerepel benne; ' +
                'ilyenkor cseréld le csak azt az egy szót a legvalószínűbb magyar szóra hangalak és kontextus alapján. ' +
                'Mondatszerkezetet, szórendet, fogalmazást, helyesírást NE változtass. ' +
                'Bizonytalan esetben maradj az eredeti szövegnél és valid=true. ' +
                'Példák: ' +
                '"Holnap hívjam vissza Lacit" -> valid:true, változatlan. ' +
                '"Mamóci Pametünk" -> valid:false, corrected_text:"". ' +
                '"Bo megyünk moziba" -> valid:true, corrected_text:"Ma megyünk moziba".',
            },
          ],
        },
        {
          role: 'user',
          content: [{ type: 'input_text', text: inputText }],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'memo_ai_check_result_v3',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              valid: { type: 'boolean' },
              corrected_text: { type: 'string' },
            },
            required: ['valid', 'corrected_text'],
          },
        },
      },
    });

    let parsed = null;

    try {
      parsed = JSON.parse(response.output_text || '{}');
    } catch (parseError) {
      console.error('Failed to parse ai-check response:', parseError);
      throw new Error('Az ai-check válasz nem volt értelmezhető JSON.');
    }

    const valid = parsed.valid !== false; // default true on parse failure
    const correctedText = valid ? normalizeText(parsed.corrected_text || inputText) : '';

    console.log('[relay-basic] /api/ai-check-memo result:', { valid, corrected_text: correctedText });

    res.json({ ok: true, valid, corrected_text: correctedText });
  } catch (error) {
    console.error('Failed to ai-check memo:', error);

    res.status(500).json({
      ok: false,
      error: 'Failed to ai-check memo',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.post('/api/query-memos', async (req, res) => {
  try {
    const { query = '', memos = [], now = new Date().toISOString() } = req.body ?? {};
    const cleanQuery = normalizeText(query);
    const nowDate = new Date(now);
    const nowMs = isNaN(nowDate.getTime()) ? Date.now() : nowDate.getTime();

    const safeMemos = Array.isArray(memos)
      ? memos
          .filter((m) => m && typeof m.text === 'string')
          .map((m) => {
            const createdMs = m.createdAt ? new Date(m.createdAt).getTime() : nowMs;
            const ageDays = Math.max(0, Math.floor((nowMs - createdMs) / (1000 * 60 * 60 * 24)));
            return {
              id: String(m.id ?? ''),
              text: normalizeText(m.text),
              done: m.done === true,
              doneAt: m.doneAt ?? null,
              ageDays,
            };
          })
          .filter((m) => m.id && m.text)
      : [];

    // Detect if the query is asking about completed items
    const queryLower = cleanQuery.toLowerCase();
    const isDoneQuery = [
      'kész', 'elintézett', 'megcsináltam', 'befejezet', 'elvégzet',
      'amit már', 'lezárt', 'kész teendő',
    ].some((kw) => queryLower.includes(kw));

    const filteredMemos = safeMemos.filter((m) => m.done === isDoneQuery);

    if (!cleanQuery) {
      res.status(400).json({ ok: false, error: 'Missing query', message: 'A kérdés hiányzik.' });
      return;
    }

    console.log('[relay-query] POST /api/query-memos:', {
      query: cleanQuery,
      totalMemos: safeMemos.length,
      filteredCount: filteredMemos.length,
      isDoneQuery,
    });

    const memoList = filteredMemos.map((m) => ({
      id: m.id,
      text: m.text,
      kor: `${m.ageDays} nap`,
    }));

    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [
            {
              type: 'input_text',
              text:
                'Te egy személyes memo appban segítesz prioritizálni és lekérdezni a hangból rögzített feljegyzéseket. ' +
                'Nincs tárolt metaadat — csak a memo szövege és kora (napokban) az alapja minden döntésnek.\n\n' +
                'FELADATOD:\n' +
                '– Ha a kérdés ÁTTEKINTÉST kér (pl. "mi a dolgom ma?", "mik a teendőim?", "mi sürgős?", ' +
                '"foglald össze mit kell tennem", "prioritizáld", "mivel kellene foglalkoznom"), ' +
                'válaszolj response_type="prioritized"-del és töltsd ki a groups tömböt.\n' +
                '– Lista kérésnél (pl. "sorold fel", "mutasd meg", "milyen", "melyik"): response_type="list", ' +
                'memo_ids az illeszkedő id-k.\n' +
                '– Minden más specifikus kérdésnél: response_type="text", text_answer-ben rövid válasz.\n\n' +
                'PRIORITIZÁLÁS — 4 szint:\n\n' +
                '🔴 urgent (SÜRGŐS — ég a körmödre):\n' +
                '  – Explicit közeli határidő: ma, holnap, holnapután, "péntekig", konkrét közeli dátum, "héten belül"\n' +
                '  – Erős sürgető megfogalmazás: "azonnal", "sürgős", "muszáj", "nagyon várják", "ne felejtsem el ma"\n' +
                '  – VAGY: 14+ napos ÉS időérzékeny tartalom (valaki vár rá, közelgő esemény, lejáró határidő)\n\n' +
                '🟡 important (FONTOS, DE VÁR):\n' +
                '  – Van konkrét teendő-szándék, de nincs közeli határidő és nem sürgető\n' +
                '  – Pl. "hívjam fel Lacit", "el kell intézni a fogszabályozót"\n\n' +
                '⚪ later (RÁÉR / ÖTLET):\n' +
                '  – "Valamikor", "jó lenne", ötletek, hosszú távú tervek, semmi sürgető\n' +
                '  – Pl. "megnézni azt a filmet amit Anna ajánlott"\n\n' +
                '🔵 stale (ROHAD — nézd meg):\n' +
                '  – 14+ napos, a tartalom alapján teendő (nem ötlet), de NEM időérzékeny vagy sürgős\n' +
                '  – Érdemes rákérdezni: még aktuális-e?\n\n' +
                'PÉLDÁK:\n' +
                '  – "Holnap reggel hívjam fel a könyvelőt" → urgent (konkrét közeli idő)\n' +
                '  – "Befizetni a biztosítást péntekig" → urgent (határidő)\n' +
                '  – "El kell menni fogorvoshoz" + kor 3 nap → important\n' +
                '  – "Válaszolni kéne Kovács úrnak" + kor 20 nap → stale\n' +
                '  – "Megnézni azt a könyvet amit Anna ajánlott" → later\n' +
                '  – "Jó lenne megtanulni gitározni" → later\n\n' +
                'SZABÁLYOK:\n' +
                '  – Csak a megadott memo-kra hivatkozz, ne találj ki semmit\n' +
                '  – groups-ban csak nem üres csoportok szerepeljenek; sorrendjük: urgent, important, stale, later\n' +
                '  – text_answer: rövid, természetes, TTS-barát összefoglaló (autóban hallgatva is érthető); ' +
                'prioritized esetén a sürgős elemeket sorolja fel elsőként; text esetén is töltsd ki; ' +
                'list esetén null\n' +
                '  – Ha nincs egyetlen memo sem, jelezd szövegesen a text_answer-ben',
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'input_text',
              text:
                `Kérdés/feladat: "${cleanQuery}"\n\n` +
                `Feljegyzések:\n${JSON.stringify(memoList, null, 2)}`,
            },
          ],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'memo_query_result_v2',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              response_type: { type: 'string', enum: ['text', 'list', 'prioritized'] },
              text_answer: { type: ['string', 'null'] },
              memo_ids: {
                type: 'array',
                items: { type: 'string' },
              },
              groups: {
                type: 'array',
                items: {
                  type: 'object',
                  additionalProperties: false,
                  properties: {
                    level: { type: 'string', enum: ['urgent', 'important', 'later', 'stale'] },
                    memo_ids: { type: 'array', items: { type: 'string' } },
                    summary: { type: 'string' },
                  },
                  required: ['level', 'memo_ids', 'summary'],
                },
              },
            },
            required: ['response_type', 'text_answer', 'memo_ids', 'groups'],
          },
        },
      },
    });

    let parsed = null;

    try {
      parsed = JSON.parse(response.output_text || '{}');
    } catch (parseError) {
      console.error('Failed to parse query-memos response:', parseError);
      throw new Error('A query-memos válasz nem volt értelmezhető JSON.');
    }

    const groups = Array.isArray(parsed.groups)
      ? parsed.groups.map((g) => ({
          level: g.level,
          memo_ids: Array.isArray(g.memo_ids) ? g.memo_ids : [],
          summary: typeof g.summary === 'string' ? g.summary : '',
        }))
      : [];

    const result = {
      ok: true,
      response_type: parsed.response_type ?? 'text',
      text_answer: parsed.text_answer ?? null,
      memo_ids: Array.isArray(parsed.memo_ids) ? parsed.memo_ids : [],
      groups,
    };

    console.log('[relay-query] /api/query-memos result:', {
      response_type: result.response_type,
      isDoneQuery,
      filteredCount: filteredMemos.length,
      groupSizes: groups.map((g) => `${g.level}:${g.memo_ids.length}`).join(', ') || '(none)',
    });

    res.json(result);
  } catch (error) {
    console.error('Failed to query memos:', error);

    res.status(500).json({
      ok: false,
      error: 'Failed to query memos',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.post('/api/apply-memo-modification', async (req, res) => {
  try {
    const { originalText = '', instruction = '' } = req.body ?? {};
    const cleanOriginal = normalizeText(originalText);
    const cleanInstruction = normalizeText(instruction);

    console.log('[relay-modify] POST /api/apply-memo-modification:', {
      originalText: cleanOriginal,
      instruction: cleanInstruction,
    });

    if (!cleanOriginal) {
      res.status(400).json({ ok: false, error: 'Missing originalText', message: 'Az eredeti szöveg hiányzik.' });
      return;
    }

    if (!cleanInstruction) {
      res.status(400).json({ ok: false, error: 'Missing instruction', message: 'A módosítási utasítás hiányzik.' });
      return;
    }

    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [
            {
              type: 'input_text',
              text:
                'Te egy precíz magyar szövegszerkesztő vagy, aki memó-alkalmazásban módosítja a mentett szövegeket.\n\n' +
                'Kapsz egy EREDETI SZÖVEGET és egy MÓDOSÍTÁSI UTASÍTÁST.\n\n' +
                'A módosítási utasítás egy hangbemondás szövege – ez lehet természetes, akár redundáns, ' +
                'ismétlő vagy magyarázó megfogalmazás (pl. "Holnap megyünk moziba, nem ma"). ' +
                'Belőle ki kell olvasni a SZÁNDÉKOLT VÁLTOZTATÁST, és azt alkalmazni az eredeti szövegre.\n\n' +
                'ALAPSZABÁLY: az eredeti szöveget a LEHETŐ LEGKISEBB MÉRTÉKBEN módosítsd. ' +
                'Csak azt változtasd, ami a módosítás szándéka szerint tényleg megváltozott. ' +
                'Ne vedd át az utasítás felesleges részeit, magyarázatait, ismétléseit.\n\n' +
                'Példák:\n' +
                '- Eredeti: "Ma megyünk moziba" | Utasítás: "Holnap megyünk moziba, nem ma" → Eredmény: "Holnap megyünk moziba"\n' +
                '- Eredeti: "Holnap délután hívjam vissza Lacit" | Utasítás: "Holnap délelőtt" → Eredmény: "Holnap délelőtt hívjam vissza Lacit"\n' +
                '- Eredeti: "Ma délután megyek moziba" | Utasítás: "holnap délután" → Eredmény: "Holnap délután megyek moziba"\n' +
                '- Eredeti: "Vegyek kenyeret és tejet" | Utasítás: "Vajat is vegyek, ne csak kenyeret és tejet" → Eredmény: "Vegyek kenyeret, tejet és vajat"\n\n' +
                'Csak a módosított szöveget add vissza, semmilyen magyarázat, prefix vagy kommentár nélkül.',
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'input_text',
              text: `Eredeti szöveg: "${cleanOriginal}"\n\nMódosítás: "${cleanInstruction}"`,
            },
          ],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'memo_modification_result',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              modified_text: { type: 'string' },
            },
            required: ['modified_text'],
          },
        },
      },
    });

    let parsed = null;

    try {
      parsed = JSON.parse(response.output_text || '{}');
    } catch (parseError) {
      console.error('[relay-modify] JSON parse error:', parseError);
      throw new Error('A módosítás válasza nem volt értelmezhető JSON.');
    }

    const modifiedText = normalizeText(parsed.modified_text ?? '');

    console.log('[relay-modify] result:', { modifiedText });

    if (!modifiedText) {
      res.status(500).json({ ok: false, error: 'Empty result', message: 'A módosítás eredménye üres.' });
      return;
    }

    res.json({ ok: true, modified_text: modifiedText });
  } catch (error) {
    console.error('Failed to apply memo modification:', error);
    res.status(500).json({
      ok: false,
      error: 'Failed to apply memo modification',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.post('/api/interpret-command-audio', upload.single('file'), async (req, res) => {
  const uploadedFile = req.file;

  try {
    if (!uploadedFile) {
      res.status(400).json({
        ok: false,
        error: 'Missing file',
        message: 'Nem érkezett hangfájl a kérésben.',
      });
      return;
    }

    const {
      language = 'hu',
      memoText = '',
      selectedCategoryId = null,
      categories = '[]',
    } = req.body ?? {};

    let parsedCategories = [];

    try {
      parsedCategories = JSON.parse(categories);
    } catch {
      parsedCategories = [];
    }

    const safeCategories = safeCategoryList(parsedCategories);
    const cleanMemoText = normalizeText(memoText);

    const stat = await fsp.stat(uploadedFile.path);

    console.log('[relay-command] POST /api/interpret-command-audio file:', {
      language,
      originalname: uploadedFile.originalname,
      mimetype: uploadedFile.mimetype,
      size: uploadedFile.size,
      statSize: stat.size,
      memoText: cleanMemoText,
      selectedCategoryId,
      categories: safeCategories,
    });

    if (stat.size <= 0) {
      res.status(400).json({
        ok: false,
        error: 'Empty file',
        message: 'A feltöltött hangfájl üres.',
      });
      return;
    }

    const transcription = await client.audio.transcriptions.create({
      file: fs.createReadStream(uploadedFile.path),
      model: 'gpt-4o-transcribe',
      language,
      response_format: 'json',
      prompt:
        'Magyar rövid hangparancs egy memo appban. ' +
        'Lehetséges parancsok: mentsd el, mentés, jó így mentsd el, rendben mentsd el, ' +
        'újra mondom, újramondom, kezdem újra, nem jó, mégsem, ' +
        'tedd az Emlékeztetők kategóriába, tedd az Egyéb megjegyzendők kategóriába, ' +
        'vagy egy felhasználói kategória neve.',
    });

    const rawCommandText = normalizeText(transcription.text);

    console.log('[relay-command] command transcription:', {
      rawCommandText,
    });

    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [
            {
              type: 'input_text',
              text:
                'Egy magyar voice-first memo app rövid hangparancsát értelmezed. ' +
                'A bemenet egy STT által felismert rövid szöveg, ami gyakran hibás lehet. ' +
                'A user valószínűleg magyarul mondott egy nagyon rövid parancsot. ' +
                'A feladatod NEM a szöveg javítása, hanem az akció felismerése. ' +
                'Csak ezek az akciók léteznek: save, retry, change_category, none. ' +
                'save, ha a szöveg akár fonetikusan is hasonlít ezekre: ' +
                '"mentsd el", "mentés", "elmentem", "jó így", "jó így mentsd el", ' +
                '"rendben", "rendben mentsd el", "oké", "mehet", "ez jó", "jó lesz". ' +
                'Nagyon fontos: ha az STT olyasmit ír, hogy "Mensch L", "ments L", "mensdel", ' +
                '"bözler", "józle", "jó lesz", "jó így", akkor ez nagy valószínűséggel save. ' +
                'retry, ha a szöveg akár fonetikusan is hasonlít ezekre: ' +
                '"újra mondom", "újramondom", "kezdem újra", "nem jó", "töröld", "másikat mondok". ' +
                'change_category, ha a user kategóriát nevez meg vagy azt mondja, hogy "tedd ... kategóriába". ' +
                'Kategóriaváltásnál csak a megadott kategórialistából választhatsz. ' +
                'none csak akkor legyen, ha tényleg nem lehet eldönteni. ' +
                'Ne magyarázz, csak JSON-t adj vissza.',
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'input_text',
              text:
                `Memo szöveg:\n${cleanMemoText}\n\n` +
                `Jelenlegi kategória id:\n${String(selectedCategoryId ?? '')}\n\n` +
                `Hangparancs STT:\n${rawCommandText}\n\n` +
                `Kategóriák:\n${JSON.stringify(safeCategories, null, 2)}`,
            },
          ],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'audio_command_interpretation',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              action: {
                type: 'string',
                enum: ['save', 'retry', 'change_category', 'none'],
              },
              category_id: {
                type: ['string', 'null'],
              },
              category_name: {
                type: ['string', 'null'],
              },
              heard_text: {
                type: 'string',
              },
            },
            required: ['action', 'category_id', 'category_name', 'heard_text'],
          },
        },
      },
    });

    let parsed = null;

    try {
      parsed = JSON.parse(response.output_text || '{}');
    } catch (parseError) {
      console.error('Failed to parse audio command interpretation response:', parseError);
      throw new Error('A hangparancs értelmezés válasza nem volt értelmezhető JSON.');
    }

    const action = parsed.action ?? 'none';

    const categoryExists = safeCategories.some(
      (category) => category.id === parsed.category_id
    );

    const result = {
      ok: true,
      action,
      category_id:
        action === 'change_category' && categoryExists
          ? parsed.category_id
          : null,
      category_name:
        action === 'change_category' && categoryExists
          ? parsed.category_name
          : null,
      heard_text: normalizeText(parsed.heard_text || rawCommandText),
      raw_transcript: rawCommandText,
    };

    console.log('[relay-command] /api/interpret-command-audio result:', result);

    res.json(result);
  } catch (error) {
    console.error('Failed to interpret command audio:', error);

    res.status(500).json({
      ok: false,
      error: 'Failed to interpret command audio',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  } finally {
    if (uploadedFile?.path) {
      try {
        await fsp.unlink(uploadedFile.path);
      } catch (cleanupError) {
        console.error('Failed to delete uploaded temp file:', cleanupError);
      }
    }
  }
});

app.post('/api/match-command', async (req, res) => {
  try {
    const { heard = '', commands = [] } = req.body ?? {};

    const cleanHeard = normalizeText(heard);
    const safeCommands = Array.isArray(commands)
      ? commands.filter((c) => typeof c === 'string' && c.trim())
      : [];

    console.log('[relay-cmd] POST /api/match-command:', { heard: cleanHeard, commands: safeCommands });

    if (!cleanHeard || safeCommands.length === 0) {
      return res.json({ ok: true, matched: null });
    }

    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [
            {
              type: 'input_text',
              text:
                'Egy magyar voice-first app felhasználója röviden bemondott valamit. ' +
                'A feladatod eldönteni, hogy a kimondott szöveg melyik elérhető gombnak felel meg. ' +
                'A felhasználó mondhat rövidebb vagy hosszabb változatot is, pl. "elmentsem" = "Elmentem". ' +
                'Ha nem illik egyik gombhoz sem, vagy nem egyértelmű, a matched értéke legyen null. ' +
                'Fontos: a matched értéknek PONTOSAN meg kell egyeznie az egyik gomb feliratával.',
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'input_text',
              text:
                `Elérhető gombok: ${JSON.stringify(safeCommands)}\n` +
                `A felhasználó ezt mondta: "${cleanHeard}"`,
            },
          ],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'command_match',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              matched: { type: ['string', 'null'] },
            },
            required: ['matched'],
          },
        },
      },
    });

    let parsed = null;
    try {
      parsed = JSON.parse(response.output_text || '{}');
    } catch {
      parsed = { matched: null };
    }

    const matched =
      typeof parsed.matched === 'string' && safeCommands.includes(parsed.matched)
        ? parsed.matched
        : null;

    console.log('[relay-cmd] /api/match-command result:', { matched });

    res.json({ ok: true, matched });
  } catch (error) {
    console.error('Failed to match command:', error);
    res.status(500).json({ ok: false, matched: null });
  }
});

app.post('/api/match-command-audio', upload.single('file'), async (req, res) => {
  const uploadedFile = req.file;

  try {
    if (!uploadedFile) {
      return res.status(400).json({ ok: false, matched: null, heard_text: null, error: 'Missing file' });
    }

    let safeCommands = [];
    try {
      const parsed = JSON.parse(req.body?.commands ?? '[]');
      safeCommands = Array.isArray(parsed)
        ? parsed.filter((c) => typeof c === 'string' && c.trim())
        : [];
    } catch {
      safeCommands = [];
    }

    if (safeCommands.length === 0) {
      return res.json({ ok: true, matched: null, heard_text: null });
    }

    const stat = await fsp.stat(uploadedFile.path);

    console.log('[relay-cmd] POST /api/match-command-audio:', {
      commands: safeCommands,
      filename: uploadedFile.originalname,
      size: stat.size,
    });

    if (stat.size < 8000) {
      return res.json({ ok: true, matched: null, heard_text: null });
    }

    // Step 1: STT — prompt biases the model toward the known command labels
    const cmdPrompt = `Magyar rövid hangparancs. Lehetséges: ${safeCommands.join(', ')}.`;

    const transcription = await client.audio.transcriptions.create({
      file: fs.createReadStream(uploadedFile.path),
      model: 'gpt-4o-transcribe',
      language: 'hu',
      response_format: 'json',
      prompt: normalizeText(cmdPrompt),
    });

    const heardText = normalizeText(transcription.text);

    console.log('[relay-cmd] /api/match-command-audio STT:', { heard: heardText });

    if (!heardText) {
      return res.json({ ok: true, matched: null, heard_text: null });
    }

    // Step 2: calibrated semantic matching
    const matchResponse = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [
            {
              type: 'input_text',
              text:
                'Te egy magyar hangparancs-azonosító vagy. ' +
                'A hangfelismerő szöveget kapsz, ami a felhasználó által kimondott parancs átirata. ' +
                'Döntsd el, melyik elérhető parancsnak felel meg. ' +
                'Kisebb kiejtési vagy ragozási eltérést fogadj el: ' +
                '"elmentsem" → "Elmentem", "mégsem" → "Mégsem", "mondok" → "Mondom". ' +
                'Ha viszont a szöveg egyértelműen nem hasonlít egyetlen parancshoz sem — ' +
                'pl. véletlenszerű szó, értelmetlen zaj-átirat, teljesen más témájú szó — ' +
                'adj null-t vissza. Kétség esetén is null-t adj. ' +
                'A matched értéknek PONTOSAN egyeznie kell az egyik parancs szövegével.',
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'input_text',
              text:
                `Elérhető parancsok: ${JSON.stringify(safeCommands)}\n` +
                `Hangfelismerő szövege: "${heardText}"`,
            },
          ],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'command_match',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              matched: { type: ['string', 'null'] },
            },
            required: ['matched'],
          },
        },
      },
    });

    let parsed = { matched: null };
    try {
      parsed = JSON.parse(matchResponse.output_text || '{}');
    } catch {
      // keep defaults
    }

    const matched =
      typeof parsed.matched === 'string' && safeCommands.includes(parsed.matched)
        ? parsed.matched
        : null;

    console.log('[relay-cmd] /api/match-command-audio result:', { matched, heard_text: heardText });

    res.json({ ok: true, matched, heard_text: heardText });
  } catch (error) {
    console.error('Failed to match command audio:', error);
    res.status(500).json({ ok: false, matched: null, heard_text: null, error: error.message });
  } finally {
    if (uploadedFile?.path) {
      try {
        await fsp.unlink(uploadedFile.path);
      } catch {}
    }
  }
});

app.post('/api/tts', async (req, res) => {
  const { text, voice = 'marin' } = req.body ?? {};

  if (typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ ok: false, error: 'Missing text' });
  }

  const TTS_PARAMS = {
    model: 'gpt-4o-mini-tts',
    input: text.trim(),
    response_format: 'mp3',
    speed: 1.2,
    instructions: 'Speak in a natural, warm, conversational Hungarian tone — like a friendly assistant, not a robot. Keep a brisk, confident pace. Use clearly rising intonation for sentences that end with a question mark.',
  };

  async function createSpeech(useVoice) {
    return client.audio.speech.create({ ...TTS_PARAMS, voice: useVoice });
  }

  function isVoiceError(error) {
    const msg = (error?.message ?? '').toLowerCase();
    return msg.includes('voice') || msg.includes('invalid_request_error');
  }

  try {
    let mp3;
    try {
      mp3 = await createSpeech(voice);
    } catch (error) {
      if (isVoiceError(error)) {
        console.warn(`[TTS] Voice '${voice}' not available, retrying with 'coral'. Error: ${error.message}`);
        mp3 = await createSpeech('coral');
      } else {
        throw error;
      }
    }

    const buffer = Buffer.from(await mp3.arrayBuffer());
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Length', buffer.length);
    res.send(buffer);
  } catch (error) {
    console.error('TTS error:', error);
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/interpret-memo-action', async (req, res) => {
  const { memoText, userInstruction } = req.body ?? {};

  if (typeof memoText !== 'string' || typeof userInstruction !== 'string') {
    return res.status(400).json({ ok: false, error: 'Missing memoText or userInstruction' });
  }

  try {
    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [{
            type: 'input_text',
            text:
              'Te egy magyar memó-asszisztens vagy. A felhasználó bemondta a memóját, amit felolvastunk nekik, ' +
              'és most reagált. Az input automatikus hangfelismerőből (STT) érkezik, ezért tartalmazhat ' +
              'kiejtési hibát — pl. "menj el" valójában "mentsd el" lehet (hasonlóan hangzik).\n\n' +
              'Döntsd el, mit szeretne:\n' +
              '- "save": el akarja menteni (pl. "mentsd el", "mentsed", "elmentem", "igen", "rendben", "ok", ' +
              '"jó", "legyen", "legyen így", "jó lesz" — és ASR-hiba esetén: "menj el", "menta el", "mentsük")\n' +
              '- "cancel": el akarja vetni (pl. "mégsem", "nem kell", "ne mentsd", "töröld", "eldobom", ' +
              '"vissza", "hagyd el" — de NEM "mentsd el" és nem "menj el")\n' +
              '- "repeat": meg szeretné hallani újra a memó szövegét (pl. "ismételd meg", "mondd el újra", ' +
              '"nem hallottam", "még egyszer", "újra", "ismételd", "mit mondtál")\n' +
              '- "modify": módosítani akarja — alkalmazd az utasítást a memó szövegére és add vissza\n\n' +
              'Fontos: ha a szöveg menti/mentsd/elment szóra hasonlít (bármilyen ragozásban), azt "save"-nek ' +
              'értelmezd. A "menj el" szinte biztosan "mentsd el" STT-hiba. ' +
              'Csak akkor adj "cancel"-t, ha egyértelműen elvetést jelent.',
          }],
        },
        {
          role: 'user',
          content: [{
            type: 'input_text',
            text: `Memó szövege: "${memoText}"\n\nFelhasználó reakciója: "${userInstruction}"`,
          }],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'memo_action',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              action: { type: 'string', enum: ['save', 'cancel', 'repeat', 'modify'] },
              new_text: { type: ['string', 'null'] },
            },
            required: ['action', 'new_text'],
          },
        },
      },
    });

    let parsed = { action: 'modify', new_text: null };
    try { parsed = JSON.parse(response.output_text || '{}'); } catch {}

    if (!['save', 'cancel', 'repeat', 'modify'].includes(parsed.action)) parsed.action = 'modify';

    console.log('[relay] /api/interpret-memo-action:', { action: parsed.action, instruction: userInstruction });
    res.json({ ok: true, action: parsed.action, new_text: parsed.new_text ?? null });
  } catch (error) {
    console.error('interpret-memo-action error:', error);
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/extract-shopping-items', async (req, res) => {
  const { memos } = req.body ?? {};

  if (!Array.isArray(memos) || memos.length === 0) {
    return res.json({ ok: true, items: [] });
  }

  const noteMemos = memos.filter((m) => m.memoType !== 'shopping_list');
  if (noteMemos.length === 0) return res.json({ ok: true, items: [] });

  const memoContext = noteMemos.map((m) => {
    const alreadyDone = m.strikethrough?.length
      ? `\n  [Már elintézett: ${m.strikethrough.join(', ')}]`
      : '';
    return `Memo [${m.id}]: "${m.text}"${alreadyDone}`;
  }).join('\n\n');

  try {
    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [{
            type: 'input_text',
            text:
              'Te egy magyar bevásárló asszisztens vagy. Az alábbiakban memók listája következik. ' +
              'Keresd meg az összes olyan terméket, élelmiszert, tárgyat, aminek megvásárlása szükséges.\n\n' +
              'Egy tétel "vásárlandó objektum" ha:\n' +
              '1. A memóban kifejezetten szerepel, hogy meg kell venni (pl. "venni kell", "vedd meg", "kell belőle", "szükség van rá")\n' +
              '2. VAGY: a memó élelmiszereket, háztartási cikkeket, gyógyszereket, vagy egyéb boltban kapható\n' +
              '   termékeket sorol fel amelyeket tipikusan bevásárláskor szoktak megvenni\n\n' +
              'NE vedd fel:\n' +
              '- Amit "Már elintézett" jelöléssel ellátott\n' +
              '- Amit egyértelműen nem lehet megvásárolni (pl. telefonhívás, dokumentum, teendő, személy)\n\n' +
              'A "text" mező legyen rövid, természetes magyar nyelvű megnevezés (pl. "tömítés", "tej 2L", "paracetamol").\n' +
              'A "sourcePhrase" a memóban szereplő PONTOS szövegrészlet, ami azonosítja a tételt — ' +
              'pontosan ugyanígy kell szerepelnie a memóban, hogy áthúzással jelölhessük.\n\n' +
              'Válaszolj JSON-ban.',
          }],
        },
        {
          role: 'user',
          content: [{ type: 'input_text', text: memoContext }],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'shopping_items',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              items: {
                type: 'array',
                items: {
                  type: 'object',
                  additionalProperties: false,
                  properties: {
                    text: { type: 'string' },
                    sourceMemoId: { type: 'string' },
                    sourcePhrase: { type: 'string' },
                  },
                  required: ['text', 'sourceMemoId', 'sourcePhrase'],
                },
              },
            },
            required: ['items'],
          },
        },
      },
    });

    const parsed = JSON.parse(response.output_text || '{"items":[]}');
    const items = Array.isArray(parsed.items) ? parsed.items : [];
    console.log('[relay] /api/extract-shopping-items:', { count: items.length });
    res.json({ ok: true, items });
  } catch (error) {
    console.error('extract-shopping-items error:', error);
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/interpret-cancel-confirm', async (req, res) => {
  const { userSpeech } = req.body ?? {};

  if (typeof userSpeech !== 'string' || !userSpeech.trim()) {
    return res.status(400).json({ ok: false, error: 'Missing userSpeech' });
  }

  try {
    const completion = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0,
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content:
            'Te egy magyar hangasszisztens vagy. Megkérdeztük a felhasználót: "Biztosan eldobod?" — ' +
            'most reagált. Az input STT-ből érkezik, tartalmazhat felismerési hibát.\n\n' +
            'Döntsd el:\n' +
            '- "confirm": egyértelműen el akarja dobni (pl. "igen", "eldobom", "töröld", "töröljük", ' +
            '"rendben", "ok", "legyen", "dobd el", "igen dobd el")\n' +
            '- "deny": nem akarja eldobni, visszalép (pl. "nem", "mégsem", "ne dobd el", "visszalépek", ' +
            '"inkább nem", "hagyjad", "mégse", "ne", "nem kell")\n\n' +
            'Fontos: az alapértelmezett válasz legyen "deny". Csak akkor adj "confirm"-et, ha a válasz ' +
            'egyértelműen megerősítő. Kétség esetén mindig "deny".\n\n' +
            'Válaszolj JSON-ban: { "action": "confirm" | "deny" }',
        },
        { role: 'user', content: userSpeech.trim() },
      ],
    });

    const raw = completion.choices[0]?.message?.content ?? '{}';
    const parsed = JSON.parse(raw);
    const action = parsed.action === 'confirm' ? 'confirm' : 'deny';
    console.log('[relay] /api/interpret-cancel-confirm:', { action, speech: userSpeech });
    res.json({ ok: true, action });
  } catch (error) {
    console.error('interpret-cancel-confirm error:', error);
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post('/api/refine-query', async (req, res) => {
  try {
    const { previousQuery = '', newSpeech = '' } = req.body ?? {};
    const cleanPrev = normalizeText(previousQuery);
    const cleanNew = normalizeText(newSpeech);

    if (!cleanNew) {
      return res.json({ ok: true, refinedQuery: cleanPrev });
    }
    if (!cleanPrev) {
      return res.json({ ok: true, refinedQuery: cleanNew });
    }

    console.log('[relay-refine] POST /api/refine-query:', { previousQuery: cleanPrev, newSpeech: cleanNew });

    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
      input: [
        {
          role: 'system',
          content: [{
            type: 'input_text',
            text:
              'Egy felhasználónak volt egy kérdése egy hangjegyzet-apphoz. Ezután még mondott valamit. ' +
              'Döntsd el:\n' +
              '- Ha az új mondanivaló MÓDOSÍTJA az előző kérdést (pl. "inkább csak a sürgősek", ' +
              '"és a tegnapiakat is", "nem holnap, hanem ma"), add vissza egy egységes, módosított kérdést.\n' +
              '- Ha az új mondanivaló TELJESEN ÚJ kérdés (pl. "listázd a könyveket", "mi van a bevásárló listán"), ' +
              'add vissza az új kérdést.\n\n' +
              'A refinedQuery legyen rövid, természetes, egyetlen kérdés vagy kérés magyarul. ' +
              'Ne magyarázz, ne adj alternatívákat.',
          }],
        },
        {
          role: 'user',
          content: [{
            type: 'input_text',
            text: `Előző kérdés: "${cleanPrev}"\n\nÚj mondanivaló: "${cleanNew}"`,
          }],
        },
      ],
      text: {
        format: {
          type: 'json_schema',
          name: 'refined_query',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              refinedQuery: { type: 'string' },
            },
            required: ['refinedQuery'],
          },
        },
      },
    });

    const parsed = JSON.parse(response.output_text || '{}');
    const refinedQuery = normalizeText(parsed.refinedQuery || cleanNew);

    console.log('[relay-refine] result:', { refinedQuery });
    res.json({ ok: true, refinedQuery });
  } catch (error) {
    console.error('refine-query error:', error);
    res.status(500).json({ ok: false, error: error instanceof Error ? error.message : 'Unknown error' });
  }
});

// ── WebSocket: /api/realtime-stt ────────────────────────────────────────────

const wss = new WebSocketServer({ server, path: '/api/realtime-stt' });

wss.on('connection', (clientWs) => {
  const OPENAI_REALTIME_URL =
    'wss://api.openai.com/v1/realtime?model=gpt-realtime-whisper';

  const openaiWs = new WebSocket(OPENAI_REALTIME_URL, {
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
  });

  function sendToClient(obj) {
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(JSON.stringify(obj));
    }
  }

  openaiWs.on('open', () => {
    try {
      openaiWs.send(JSON.stringify({
        type: 'session.update',
        session: {
          modalities: ['text'],
          language: 'hu',
          input_audio_format: 'pcm16',
          input_audio_transcription: {
            model: 'gpt-4o-transcribe',
          },
          turn_detection: {
            type: 'server_vad',
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
          },
        },
      }));
    } catch (err) {
      console.error('[realtime-stt] Failed to send session.update:', err);
    }
  });

  openaiWs.on('message', (data) => {
    try {
      const event = JSON.parse(data.toString());
      switch (event.type) {
        case 'conversation.item.input_audio_transcription.delta':
          sendToClient({ type: 'transcript_delta', delta: event.delta ?? '' });
          break;
        case 'conversation.item.input_audio_transcription.completed':
          sendToClient({ type: 'transcript_done', text: event.transcript ?? '' });
          break;
        case 'input_audio_buffer.speech_started':
          sendToClient({ type: 'speech_started' });
          break;
        case 'input_audio_buffer.speech_stopped':
          sendToClient({ type: 'speech_stopped' });
          break;
        case 'error':
          sendToClient({ type: 'error', message: event.error?.message ?? 'OpenAI error' });
          break;
        default:
          break;
      }
    } catch (err) {
      console.error('[realtime-stt] Failed to handle OpenAI message:', err);
    }
  });

  openaiWs.on('error', (err) => {
    console.error('[realtime-stt] OpenAI WebSocket error:', err);
    sendToClient({ type: 'error', message: 'Failed to connect to OpenAI' });
    clientWs.close();
  });

  openaiWs.on('close', () => {
    sendToClient({ type: 'error', message: 'OpenAI connection lost' });
    clientWs.close();
  });

  clientWs.on('message', (data, isBinary) => {
    try {
      if (isBinary || Buffer.isBuffer(data)) {
        if (openaiWs.readyState === WebSocket.OPEN) {
          const b64 = (Buffer.isBuffer(data) ? data : Buffer.from(data)).toString('base64');
          openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: b64 }));
        }
        return;
      }

      const msg = JSON.parse(data.toString());
      if (openaiWs.readyState !== WebSocket.OPEN) return;

      if (msg.type === 'commit') {
        openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
      } else if (msg.type === 'clear') {
        openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.clear' }));
      } else if (msg.type === 'ping') {
        sendToClient({ type: 'pong' });
      } else if (msg.type === 'audio_chunk') {
        openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: msg.data }));
      }
    } catch (err) {
      console.error('[realtime-stt] Failed to handle client message:', err);
    }
  });

  clientWs.on('close', () => {
    if (openaiWs.readyState === WebSocket.OPEN || openaiWs.readyState === WebSocket.CONNECTING) {
      openaiWs.close();
    }
  });
});

// ────────────────────────────────────────────────────────────────────────────

server.listen(PORT, () => {
  console.log(`Relay listening on http://localhost:${PORT}`);
});