# bsc-vanity

Tool tìm địa chỉ ví **BSC / EVM** có số đẹp (ví dụ `0x...66666666`), chạy trên **GPU qua OpenCL**, viết bằng Rust.

Trên RTX 4060 Ti đo được khoảng **600–660 triệu key/s**.

## Yêu cầu

- Rust (cài qua [rustup](https://rustup.rs)).
- GPU có driver hỗ trợ OpenCL. Với NVIDIA/AMD/Intel thì driver thường đã có sẵn, **không cần cài CUDA hay OpenCL SDK**, vì tool tự nạp `OpenCL.dll` lúc chạy.

## Build

```
cargo build --release
```

Exe nằm ở `target\release\bsc-vanity.exe`. Exe link tĩnh C runtime (cấu hình trong `.cargo/config.toml`), nên chép sang máy Windows khác vẫn chạy được mà không cần VC++ Redistributable.

## Cách dùng

```
bsc-vanity.exe -s 66666666              # đuôi
bsc-vanity.exe -p 888                   # đầu (sau 0x)
bsc-vanity.exe -p 888 -s 66666666       # cả đầu và đuôi
```

| Tham số | Ý nghĩa |
|---|---|
| `-s`, `--suffix HEX` | Đuôi địa chỉ |
| `-p`, `--prefix HEX` | Đầu địa chỉ (có thể kèm `0x`) |

- Chỉ nhận ký tự hex (`0-9`, `a-f`), không phân biệt hoa thường.
- Nếu **double-click** exe (không truyền tham số), tool sẽ hỏi đuôi/đầu trong cửa sổ và chờ bấm Enter trước khi đóng.

Kết quả được in ra màn hình và **ghi thêm** một dòng vào `found.txt` trong thư mục đang chạy, không ghi đè kết quả cũ. Tool in ra đường dẫn đầy đủ của file sau khi lưu:

```
Address: 0x1234...abcd66666666 | Private key: 0x...
saved to E:\...\found.txt
```

## Thời gian ước tính

Mỗi ký tự thêm vào thì lâu hơn 16 lần. Số liệu dưới đây là trung bình ở khoảng 640 M key/s:

| Số ký tự | Thời gian |
|---|---|
| 8 | ~7 giây |
| 9 | ~2 phút |
| 10 | ~30 phút |
| 11 | ~8 giờ |
| 12 | ~5 ngày |

## Bảo mật

- Key gốc mỗi lần chạy là số ngẫu nhiên **256-bit** lấy từ hệ điều hành (`OsRng`). Profanity bị hack năm 2022 vì seed chỉ 32-bit, tool này không mắc lỗi đó.
- GPU chỉ tìm. Khi GPU báo trúng, **CPU tính lại địa chỉ từ private key** và chỉ lưu nếu khớp.
- `found.txt` chứa private key **dạng chữ thường, không mã hóa**. File đã nằm trong `.gitignore`. Hãy sao lưu ra chỗ an toàn (USB hoặc trình quản lý mật khẩu) rồi xóa bản trên ổ đĩa.
- Trước khi nạp tiền, hãy import private key vào MetaMask/Trust Wallet và kiểm tra địa chỉ khớp.
- Không thể suy ngược private key từ một địa chỉ có sẵn. Tool chỉ tìm *một* ví bất kỳ khớp mẫu.

## Cách hoạt động

```
src/main.rs    host: đọc tham số, sinh key, chuẩn bị điểm xuất phát, chạy kernel, kiểm tra lại và lưu
src/kernel.cl  GPU: phép toán secp256k1, keccak256, so mẫu
```

1. CPU chọn key gốc ngẫu nhiên `k0`. Luồng GPU thứ `g` bắt đầu ở key `k0 + g·2^40`, nên các luồng không bao giờ trùng nhau.
2. Mỗi vòng, một luồng tính `P+1G … P+MG` bằng phép cộng điểm. M phép nghịch đảo được gom thành **1 lần** (Montgomery batch inversion).
3. Với mỗi điểm: tính `keccak256(x‖y)`, lấy 20 byte cuối làm địa chỉ, rồi so với mặt nạ mẫu ngay trên GPU.
4. Khi khớp, GPU trả về `(luồng, vị trí)`, CPU suy ra private key từ đó.

Các thông số chỉnh ở đầu `src/main.rs`. Bộ hiện tại đã đo là nhanh nhất trên RTX 4060 Ti:

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `M` | 128 | Số key mỗi vòng, dùng chung 1 phép nghịch đảo |
| `N` | 2^15 | Số luồng GPU |
| `ROUNDS` | 16 | Số vòng mỗi lần chạy kernel. Giữ mỗi lần chạy ≪ 2 giây để Windows không reset GPU (TDR) |

## Test

```
cargo test --release
```

- `cpu_helpers`: private key `1` phải ra đúng địa chỉ đã biết `0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf`. Test thêm phần so mẫu và checksum EIP-55.
- `gpu_finds_planted_key`: cài sẵn một key ở vị trí đã biết rồi bắt GPU tìm lại đúng key đó. Test này kiểm tra toàn bộ phép toán trên GPU và cách tính chỉ số. Cần có GPU.
