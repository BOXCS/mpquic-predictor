## 🚀 Panduan Menjalankan Aplikasi (Execution Guide)

Untuk menjalankan sistem secara keseluruhan, Anda perlu membuka **4 terminal terpisah** dan menjalankannya sesuai dengan urutan di bawah ini.

### 📌 Urutan Eksekusi & Konfigurasi Terminal

| Urutan | Komponen | Lingkungan (Environment) | Perintah (Command) |
| :---: | --- | --- | --- |
| **1** | **MP-QUIC Server** | Windows PowerShell | Menyiapkan server QUIC untuk menerima koneksi client. |
| **2** | **FastAPI Backend** | Windows PowerShell | Menyediakan API untuk manajemen dan switch path proaktif. |
| **3** | **Frontend** | WSL (Ubuntu-22.04) | Interface monitoring berbasis web. |
| **4** | **MP-QUIC Client** | Raspberry Pi (SSH/Terminal) | Mengirimkan data metrik IoT melalui MP-QUIC. |

---

### 💻 Langkah demi Langkah (Step-by-Step Execution)

#### 1. Terminal 1: MP-QUIC Server (Windows PowerShell)
> **Catatan:** Server harus berjalan terlebih dahulu sebelum Client (Raspberry Pi) mencoba melakukan koneksi.

```powershell
# Aktivasi Virtual Environment & Pindah ke direktori project di WSL
C:\mpquic-venv\Scripts\activate
cd \\wsl$\Ubuntu-22.04\home\zaky\mpquic-ai

# Jalankan server
python -m server.mpquic_server

# Aktivasi Virtual Environment & Pindah ke direktori project di WSL
C:\mpquic-venv\Scripts\activate
cd \\wsl$\Ubuntu-22.04\home\zaky\mpquic-ai

# Jalankan Uvicorn backend server
python -m uvicorn web.backend.main:app --reload --host 0.0.0.0 --port 8080

# Pindah ke direktori frontend
cd ~/mpquic-ai/web/frontend

# Jalankan development server
npm run dev

# Pindah ke direktori project di Raspi & Aktivasi venv
cd ~/mpquic-ai
source venv/bin/activate

# Jalankan client hardware
python3 -m hardware.mpquic_client
