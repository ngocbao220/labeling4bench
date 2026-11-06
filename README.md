# Labeling benchmark: turn detection tiếng Việt

Công cụ dựng bộ data benchmark cho các mô hình phát hiện lượt nói (turn detection) tiếng Việt: **Easy Turn** (utterance-level) và **SoulX-Duplug** (frame-level). Repo gồm:

- **Pipeline thu và cắt data** (`tools/`): lấy audio từ các nguồn trên server cho đến khi đủ quota theo domain và kịch bản, cắt thành segment theo sự kiện turn (nói xong / ngừng giữa chừng / tiếng đệm / chen ngang), kèm mốc thời gian để đo latency.
- **Hỏi mô hình** (Qwen3-Omni Instruct và Thinking) đúng bộ câu hỏi gán nhãn, để biết câu nào máy làm được và câu nào cần người.
- **Trang gán nhãn** (`index.html` + `server.py`) cho người gán nhãn.
- **Thống kê, audit câu hỏi và notebook** để xem trực quan kết quả.

Thiết kế câu hỏi và cách khai thác nhãn: xem [`docs/`](docs/README.md).

---

## 1. Cài đặt

```bash
git clone https://codeberg.org/ngocbao/labeling-benchmark.git
cd labeling-benchmark
pip install -r tools/requirements.txt     # trên server chạy pipeline
# cài thêm ffmpeg (apt install ffmpeg / brew install ffmpeg)
```

Chỉ chạy trang gán nhãn thì không cần cài gì, chỉ cần Python 3.

## 2. Cấu hình (`config/`, Hydra)

```
config/config.yaml            # paths: thư mục dùng chung (raw, clips, manifests, answers)
config/models/default.yaml    # đường dẫn mô hình: VAD, Nemotron-Diarization, zipformer, Qwen3-Omni…
config/collect/default.yaml   # thư mục nguồn, bucket (domain/kịch bản) và số phút cần lấy
config/segment/default.yaml   # ngưỡng gán sự kiện: bc_max, int_gap
config/manifest/default.yaml  # cửa sổ clip quanh sự kiện: pre, post
config/quotas/default.yaml    # số segment tối thiểu mỗi ô thống kê
```

Mỗi tham số chỉ nằm ở một file. Trên server, tạo bản riêng cho máy (không commit lên git):

```bash
cp config/models/default.yaml  config/models/local.yaml    # sửa đường dẫn mô hình
cp config/collect/default.yaml config/collect/local.yaml   # sửa thư mục nguồn, quota
```

Rồi thêm `models=local collect=local` vào mọi lệnh. Có thể ghi đè từng giá trị ngay trên dòng lệnh:

```bash
python3 tools/collect.py models=local collect=local collect.buckets.0.target_minutes=60 segment.bc_max=1.5
```

Giá trị ghi đè trên dòng lệnh phải viết không dấu (Hydra không nhận ký tự tiếng Việt trong override). Mô tả có dấu thì sửa trong file yaml.

**Chạy song song**, mỗi chỗ chỉnh ở đúng một khoá:

| Khoá | Ý nghĩa |
|---|---|
| `collect.workers` | số file nguồn xử lý cùng lúc trong `collect.py`: ffmpeg, ASR và gọi Qwen chạy song song; Nemotron-Diarization dùng chung GPU nên vẫn lần lượt |
| `models.<tên>.concurrency` | số request gửi cùng lúc đến vLLM trong `ask_models.py` (vLLM tự gom batch) |
| `manifest.workers` | số tiến trình ffmpeg cắt clip cùng lúc |
| `--workers` của `augment.py` | số segment biến đổi cùng lúc (mặc định bằng số nhân CPU) |

Tăng `concurrency` cho đến khi vLLM báo hết KV cache hoặc độ trễ tăng vọt. Tăng `collect.workers` cho đến khi CPU hoặc GPU đầy.

**Bucket** được xếp theo nhãn mà mô hình đoán, không theo thư mục nguồn. Ví dụ:

```yaml
- {name: interview, desc: phỏng vấn, match: {genre: [INTERVIEW]}, target_minutes: 30}
- {name: phone, desc: qua điện thoại, match: {channel: [PHONE]}, target_minutes: 30}
```

`match` dùng đúng các mã đáp án của bộ câu hỏi (`genre`, `channel`, `noise_type`, `distance`, `accent`), cộng thêm `speakers` (số người nói, đếm từ diarization).

## 3. Chạy pipeline (trên server)

Các lệnh đều có thanh tiến trình và in kết quả dạng bảng (rich) ra stderr. stdout chỉ chứa dữ liệu, ví dụ JSON của `build_manifest.py` hay báo cáo của `stats.py`, nên vẫn chuyển hướng vào file được.

**Bước 1: chạy Qwen bằng vLLM.** Lệnh in sẵn từ config:

```bash
python3 tools/conf.py models=local          # in lệnh `vllm serve ...` cho từng mô hình
```

Chạy mỗi lệnh trong một terminal (hoặc tmux) riêng.

**Bước 2: thu audio và cắt segment.**

```bash
python3 tools/collect.py models=local collect=local dry_run=true   # xem số file nguồn, quota còn thiếu
python3 tools/collect.py models=local collect=local
```

Với mỗi file nguồn, `collect.py` làm lần lượt:
1. Chuẩn hoá về wav 16 kHz.
2. Chạy VAD, diarization và ASR.
3. Cho Qwen phân loại domain và điều kiện âm thanh.
4. Nếu file khớp một bucket còn thiếu quota thì cắt thành segment, ghi vào `manifests/<id>.jsonl` và `clips/`.

Lệnh dừng khi mọi bucket đủ quota hoặc hết nguồn. Chạy lại thì tiếp tục từ chỗ dừng, và tăng `target_minutes` là lấy thêm.

**Xem quyết định LẤY / BỎ của từng file.** Lúc chạy, mỗi file in một dòng `+ LẤY` hoặc `- BỎ` kèm lý do và nhãn mô hình. Xem lại sau:

```bash
python3 tools/decisions.py collect=local                    # tổng hợp theo lý do + danh sách từng file
python3 tools/decisions.py --only BỎ --grep LECTURE         # lọc
python3 tools/decisions.py --csv out/decisions.csv          # xuất ra Excel
```

Có 5 lý do: vào bucket nào, không khớp bucket nào, khớp nhưng bucket đã đủ quota, ít tiếng nói (dưới `collect.min_speech_minutes`), mô hình không trả lời được. Mọi quyết định lưu ở `data/raw/classified.jsonl` (các trường `decision`, `reason`, `labels`, `decided_at`).

**Bước 3: hỏi mô hình bộ câu hỏi trên từng segment.**

```bash
python3 tools/ask_models.py --manifest 'manifests/*.jsonl' --model qwen_instruct --out out/qwen_instruct.jsonl models=local
python3 tools/ask_models.py --manifest 'manifests/*.jsonl' --model qwen_thinking --out out/qwen_thinking.jsonl models=local
```

Thêm `--limit 5` để chạy thử, hoặc `--dry-run` để chỉ in prompt và JSON schema. Prompt sinh tự động từ bộ câu hỏi trong `data/config.json`.

**Bước 4: thống kê độ phủ.**

```bash
python3 tools/stats.py --answers out/qwen_instruct.jsonl out/qwen_thinking.jsonl --csv out/stats collect=local
```

Kết quả gồm số phút theo bucket, phân bố từng thuộc tính, các bảng `event × domain` và `event × điều kiện âm thanh`, độ đồng thuận giữa hai mô hình, và danh sách ô chưa đạt quota.

**Bước 5 (khi thiếu điều kiện âm thanh): tạo data tổng hợp.**

```bash
python3 tools/augment.py --manifest manifests/<id>.jsonl --op noise  --noise-dir /data/musan/noise --out manifests/<id>_noise.jsonl
python3 tools/augment.py --manifest manifests/<id>.jsonl --op phone  --out manifests/<id>_phone.jsonl
python3 tools/augment.py --manifest manifests/<id>.jsonl --op reverb --rir-dir /data/rirs --out manifests/<id>_reverb.jsonl
python3 tools/augment.py --manifest manifests/<id>.jsonl --op quiet  --gain-db -20 --out manifests/<id>_quiet.jsonl
```

Segment tổng hợp có cờ `synthetic: true`, và `stats.py` báo riêng data thật và data tổng hợp.

## 4. Gán nhãn

```bash
HOST=0.0.0.0 PORT=8000 EDIT_TOKEN=matkhau python3 server.py
```

1. Mở `http://<server>:8000`. Trang tự lấy danh sách segment từ `manifests/` (những mẫu đã có clip) và đặt sẵn mốc kết thúc lượt.
2. Nhập tên người gán nhãn, nghe audio, trả lời các tab câu hỏi, kéo lại mốc trên sóng âm nếu lệch, rồi bấm xuất JSON.
3. Nút "✎ Chỉnh sửa câu hỏi" (cần `EDIT_TOKEN`) sửa câu hỏi ngay trên trang và lưu vào `data/config.json`.

Gộp các file JSON đã xuất thành một file, mỗi dòng một bản ghi:

```bash
for f in mau_*.json; do python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])),ensure_ascii=False))" "$f"; done > humans.jsonl
```

Không có server (mở thẳng `index.html` hoặc qua GitHub Pages) thì trang chỉ đọc: hiện các mẫu ví dụ trong `audio/` và không sửa được câu hỏi.

## 5. Audit câu hỏi và xem kết quả

```bash
python3 tools/audit_questions.py --humans humans.jsonl --model out/qwen_instruct.jsonl --model out/qwen_thinking.jsonl
```

Với mỗi câu hỏi, công cụ kết luận một trong bốn: **DROP** (mô hình làm tốt, bỏ khỏi phần người), **REWRITE** (người cũng không đồng thuận, câu hỏi mơ hồ), **KEEP** (cần người), **WATCH** (theo dõi thêm).

Notebook `notebooks/inspect_answers.ipynb` cho xem trực quan:
- phân bố đáp án theo từng câu và từng mô hình;
- bảng đồng thuận, ma trận nhầm lẫn;
- từng segment, gồm audio, transcript, mốc thời gian, bảng trả lời (ô bất đồng được tô màu) và phần suy nghĩ của bản Thinking.

```bash
jupyter lab notebooks/inspect_answers.ipynb
```

Khi sửa câu hỏi, sửa ở cả `index.html` (bộ mặc định) và `data/config.json` (bộ server dùng), và giữ nguyên `id` cùng mã đáp án nếu ý nghĩa không đổi.

## 6. Thử ở local không cần GPU

```bash
python3 tools/ask_models.py --manifest manifests/10154_2.jsonl --dry-run                     # xem prompt + schema
python3 tools/ask_models.py --manifest manifests/10154_2.jsonl --fake --out out/fake.jsonl   # đáp án ngẫu nhiên
python3 tools/stats.py --answers out/fake.jsonl
python3 tools/segment.py --events-only <dir>/<id>     # gán lại sự kiện từ <id>_transcript.txt có sẵn
python3 tools/collect.py dry_run=true
```

## Cấu trúc dữ liệu

| Đường dẫn | Nội dung | Commit |
|---|---|---|
| `data/config.json` | bộ câu hỏi server dùng | có |
| `manifests/*.jsonl` | mỗi dòng một segment: `event_auto`, `times` (mốc latency), `audio`, `bucket`, `file_labels`, `*_auto` | có |
| `data/raw/` | wav đã chuẩn hoá, transcript, `raw_index.jsonl`, `classified.jsonl` | không |
| `clips/` | clip của từng segment | không |
| `out/` | câu trả lời của mô hình, thống kê, log của Hydra | không |

Mốc trong `times`, tính theo giây trong clip:
- **EOT:** `t_end`.
- **MOT:** `t_pause_start`, `t_pause_end`.
- **BC:** `t_bc_start`, `t_bc_end`.
- **INT:** `t_int_start`.

## Lưu ý

- **Phần chưa chạy thử với mô hình thật:** gọi Nemotron-Diarization (NeMo), zipformer (sherpa-onnx) và gửi audio vào vLLM. Kiểm tra lần đầu với `--limit` và `collect.buckets.*.target_minutes` nhỏ.
- **Gán sự kiện tự động chỉ là gợi ý.** Chen ngang không chồng lời chỉ được bắt khi câu trước kết thúc bằng từ bỏ ngỏ ("mà", "thì", "là"…).
- **Data CSKH** phải che thông tin cá nhân trước khi đưa cho người gán nhãn.
