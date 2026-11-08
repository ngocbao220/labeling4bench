# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mục tiêu

Repo này phục vụ việc chọn data và gán nhãn để làm **benchmark turn detection cho tiếng Việt**, đánh giá hai kiểu mô hình:
- **Easy Turn**: utterance-level, phán đoán trên cả một câu/đoạn (nói xong / chưa xong / tiếng đệm / chen ngang).
- **SoulX-Duplug**: frame-level, phán đoán theo từng khung thời gian, nên cần mốc thời gian chính xác (vd. `turn_end_sec`).

Yêu cầu với data benchmark: phủ đủ domain (podcast, trò chuyện, phỏng vấn YouTube; phim; chăm sóc khách hàng…) và đủ kịch bản thực tế (nhiễu, nói nhỏ, mic gần/xa, qua điện thoại…), được **phân tầng theo độ khó** để phản ánh đúng chất lượng mô hình ngoài thực tế. Khi đổi câu hỏi hay pipeline, giữ cho nhãn đủ để lọc/phân tầng theo domain, điều kiện âm thanh và độ khó.

## Chạy

Không có build hay lint. Trang và server chỉ cần Python 3 (stdlib):

    python3 server.py                                    # http://127.0.0.1:8000
    HOST=0.0.0.0 PORT=8000 EDIT_TOKEN=matkhau python3 server.py

Mở thẳng `index.html` hoặc qua GitHub Pages (https://ngocbao220.github.io/labeling4bench/) cũng chạy được nhưng chỉ đọc: không có `/api/config`, nút sửa câu hỏi tự ẩn, trang dùng câu hỏi mặc định trong `index.html`.

Pipeline dựng data (chạy trên server có GPU; luồng đầy đủ và lệnh trong `docs/quy-trinh.md`):

    python3 tools/collect.py dry_run=true                  # config ở config/ (Hydra); trên server: models=local collect=local
    python3 tools/ask_models.py --manifest 'manifests/*.jsonl' --dry-run         # in prompt + schema, chạy được ở local
    python3 tools/ask_models.py --manifest 'manifests/*.jsonl' --fake --out out/fake.jsonl   # đáp án ngẫu nhiên để thử stats/notebook
    python3 tools/segment.py --events-only <dir>/<id>      # gán lại sự kiện từ transcript có sẵn, không cần GPU
    python3 tools/stats.py --answers out/a.jsonl out/b.jsonl
    python3 tools/audit_questions.py --humans humans.jsonl --model out/a.jsonl

Không có test suite. Kiểm tra bằng các chế độ `--dry-run` / `--fake` / `--events-only` ở trên, và chạy `notebooks/inspect_answers.ipynb` bằng `jupyter nbconvert --execute`.

## Kiến trúc

- **`index.html`**: toàn bộ app gán nhãn (HTML/CSS/JS thuần, một file). Các phần chính:
  - `SAMPLES`: danh sách mẫu, mỗi mẫu có `tags` (`turntaking` hoặc `slm`); audio ở `audio/`.
  - `PANELS`: map `tag -> [panel]`, mỗi panel là một tab gồm `qs` (câu hỏi). Tag của mẫu quyết định tab nào hiện. Câu hỏi có `codes` thì giá trị lưu là mã (vd. `EOT`, `close_mic`) thay cho chỉ số đáp án.
  - Panel `type:"score"` (thang `SCALE_*`, sinh câu chấm điểm từ `descs`) vẫn được hỗ trợ trong `buildPanels`, nhưng slm hiện không dùng.
  - Nút xuất JSON tải về một bản ghi `{sample_id, annotator, answers: {"<panel>.<qid>": value}, marks: {turn_end_sec}, missing}`. Đây cũng là định dạng đầu vào của `tools/audit_questions.py`.
- **`server.py`**: phục vụ `/`, `/index.html`, `/audio/*`, `/clips/*`, `GET /api/samples` và `GET/PUT/DELETE /api/config`. `valid()` kiểm tra cấu trúc config trước khi lưu.
- **`data/config.json`**: bộ câu hỏi đã lưu từ trình soạn trong trang. Nó **ghi đè toàn bộ** `PANELS` mặc định khi chạy có server. Vì vậy sửa câu hỏi thì phải sửa **cả** `index.html` và `data/config.json` (rồi kiểm tra bằng `server.valid`). `data/` được commit lên git (không còn trong `.gitignore`).
- **Pipeline `tools/`**: `collect.py` lấy file từ thư mục nguồn cho đến khi đủ quota theo bucket. Bucket xếp theo nhãn mô hình đoán (`match` dùng đúng `codes` của câu hỏi), **không** theo đường dẫn nguồn. Với mỗi file, `collect.py` gọi `segment.py` (VAD, diarization, ASR, quy tắc gán EoT/MoT/BC/INT), rồi `ask_models.ask` để phân loại mức file, rồi `build_manifest.build` để ra `manifests/<id>.jsonl` và `clips/`. Thứ tự ưu tiên nhãn trong `stats.py` là người > mô hình > nhãn mức file > `event_auto`.
- **`tools/ask_models.py`** là chỗ duy nhất dựng prompt và JSON schema, sinh từ `data/config.json`. `collect.py` và notebook dùng chung code này. Prompt chỉ đưa một mốc giây trung tính, vì tên các trường trong `times` (`t_pause_start`, `t_bc_start`…) sẽ lộ đáp án `event`.
- **`manifests/*.jsonl`**: mỗi dòng một segment, gồm `event_auto`, `times` (mốc đo latency theo loại sự kiện, tính theo giây trong clip), `audio` (đường dẫn clip), `bucket`, `file_labels`, và các trường `*_auto`. Mẫu tổng hợp từ `augment.py` có `synthetic: true`. Khi chạy có server, trang đọc manifest qua `GET /api/samples` (chỉ những mẫu có clip) và thay cho `SAMPLES` cứng.
- **`tools/audit_questions.py`**: so nhãn người với nhãn mô hình theo từng câu hỏi, kết luận DROP (để mô hình tự gán) / REWRITE / KEEP / WATCH.
- **Config (`config/`, Hydra)**: `config.yaml` chứa `paths` (thư mục dùng chung: raw, clips, manifests, answers) và gom các nhóm `models/` (đường dẫn mô hình), `collect/` (nguồn, bucket, quota phút, phân loại mức file), `segment/` (ngưỡng gán sự kiện), `manifest/` (cửa sổ clip) và `quotas/` (chỉ tiêu segment). Mỗi tham số chỉ nằm ở **một** file: code không có giá trị mặc định trùng với config, và các tool không có cờ dòng lệnh trùng với khoá config. `collect.py` dùng `@hydra.main`. Các tool còn lại dùng argparse, đọc config qua `tools/conf.py` (`conf.load`), và nhận ghi đè dạng `key=value` đặt ở bất kỳ đâu trên dòng lệnh (`conf.split_argv`). Cấu hình riêng của từng máy để ở `config/<nhóm>/local.yaml`.
- **Song song**: các bước tốn CPU (`collect.py`, `augment.py`) chạy **nhiều tiến trình, mỗi tiến trình 1 luồng** qua `tools/workers.py` (`pool()` + `single_thread()`), để mỗi PID ≤ 100% CPU. Tải CPU chỉ chỉnh bằng số tiến trình; zipformer và ffmpeg (`-threads 1`) cố định 1 luồng. Hàm worker phải ở cấp module (`collect.process_file`) để `spawn` pickle được. Gọi vLLM (`ask_models.py`) và cắt clip (`build_manifest.py`, mỗi lệnh ffmpeg một PID) vẫn dùng luồng, vì gần như không tốn CPU của tiến trình chính.
- **Không commit lên git**: `config/*/local.yaml`, `clips/`, `data/raw/`, `out/` (xem `.gitignore`).

## Quy ước

- Toàn bộ nội dung hiển thị và câu hỏi viết bằng tiếng Việt có dấu.
- Đổi câu hỏi thì giữ ổn định `id`/`codes` nếu được, vì chúng là khoá trong dữ liệu đã xuất và trong audit.
