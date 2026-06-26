"""
Doküman Benzerlik Karşılaştırıcı
========================================================================
1) TEXT EXTRACTION : PDF'ten metin çıkarma (pypdf) ve XLSX'ten metin
                     çıkarma (openpyxl).  Dosya uzantısına göre otomatik seçim.
2) TOKENIZER       : Sayısal-değer koruyan özel tokenizer'ı.
3) SHINGLE (n-gram): Tek kelimeler yerine ardışık kelime grupları kullanılır.
                     SEBEP: tek kelime çok kaba bir birim; aynı dildeki iki uzun
                     doküman içerikleri alakasız olsa bile binlerce ortak kelime
                     paylaşır ve yapay olarak benzer görünür. Shingle, iki
                     dokümanın ancak gerçek İFADELERİ paylaşması halinde benzer
                     çıkmasını sağlar.
4) SIMHASH         : Bağımsız 64-bit SimHash (trafilatura gerekmez).
5) SAMPLING        : Dokümanı PARÇALARA (chunk) böler, her parçayı SimHash'ler,
                     parçaları eşleştirip ortalamasını alır.

Kullanım:
    python3 comp.py rapor_a.pdf rapor_b.pdf
    python3 comp.py tablo_a.xlsx tablo_b.xlsx
    python3 comp.py rapor_a.pdf tablo_b.xlsx   # karışık format da desteklenir
"""

import sys
import os
import re
import hashlib
import time

# 0. TEXT EXTRACTION
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


def extract_from_pdf(path: str) -> dict:
    """PDF'ten metin çıkarır (tüm sayfalar birleştirilir)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return {"success": False, "error": "pypdf kurulu değil: pip install pypdf"}

    try:
        size = os.path.getsize(path)
    except OSError as e:
        return {"success": False, "error": f"Dosya bulunamadı: {e}"}

    if size > MAX_FILE_SIZE:
        limit_mb = MAX_FILE_SIZE // 1024 // 1024
        return {"success": False, "error": f"Dosya {limit_mb}MB çıkarım sınırını aşıyor"}

    try:
        reader = PdfReader(path)
        total_pages = len(reader.pages)
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(parts)
        return {"success": True, "text": text, "kind": "pdf", "unit_count": total_pages, "unit_label": "sayfa"}
    except Exception as e:
        return {"success": False, "error": f"PDF ayrıştırma hatası: {e}"}


def extract_from_xlsx(path: str) -> dict:
    """XLSX'ten metin çıkarır (tüm sayfalar/satırlar birleştirilir)."""
    try:
        import openpyxl
    except ImportError:
        return {"success": False, "error": "openpyxl kurulu değil: pip install openpyxl"}

    try:
        size = os.path.getsize(path)
    except OSError as e:
        return {"success": False, "error": f"Dosya bulunamadı: {e}"}

    if size > MAX_FILE_SIZE:
        limit_mb = MAX_FILE_SIZE // 1024 // 1024
        return {"success": False, "error": f"Dosya {limit_mb}MB çıkarım sınırını aşıyor"}

    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        parts = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None and str(c).strip()]
                if cells:
                    parts.append("\t".join(cells))
        wb.close()
        text = "\n".join(parts)
        sheet_count = len(wb.sheetnames)
        return {"success": True, "text": text, "kind": "xlsx", "unit_count": sheet_count, "unit_label": "sayfa (sekme)"}
    except Exception as e:
        return {"success": False, "error": f"XLSX ayrıştırma hatası: {e}"}


def extract_text(path: str) -> dict:
    """Dosya uzantısına göre uygun çıkarıcıyı seçer."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_from_pdf(path)
    elif ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        return extract_from_xlsx(path)
    else:
        return {"success": False, "error": f"Desteklenmeyen dosya türü: '{ext}'. Desteklenenler: .pdf, .xlsx"}


# 1. TOKENIZER — sayısal-değer koruyan tokenizer'ı
def tokenize(text: str, min_length: int = 2) -> list[str]:
    """Sayısal değerleri (15.661, 37,786) koruyarak token'lara böler."""
    tokens = re.findall(r'[a-zA-Z0-9]+(?:[.,][0-9]+)*', text.lower())
    return [t for t in tokens if len(t) >= min_length]


# 2. SHINGLE — ardışık kelime gruplarını (n-gram) üret
# ======================================================================
NGRAM = 2


def make_shingles(tokens: list[str], n: int = NGRAM) -> list[str]:
    """
    Token listesinden örtüşmeli n-gram'lar üretir.

    ["yillik","gelir","15.661","milyon"] , n=3 ->
        ["yillik gelir 15.661", "gelir 15.661 milyon"]

    Token sayısı n'den azsa, tek bir shingle olarak tüm token'ları birleştirir.
    """
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# 3. SIMHASH — bağımsız 64-bit implementasyon
# ======================================================================
HASH_BITS = 128


def _token_hash(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode()).digest(), byteorder='big')


def simhash(features: list[str], hash_bits: int = HASH_BITS) -> int:
    """Bir özellik (shingle) listesinden 64-bit SimHash parmak izi üretir."""
    if not features:
        return 0

    vector = [0] * hash_bits
    for feat in features:
        h = _token_hash(feat)
        for i in range(hash_bits):
            if (h >> i) & 1:
                vector[i] += 1
            else:
                vector[i] -= 1

    fingerprint = 0
    for i in range(hash_bits):
        if vector[i] >= 0:
            fingerprint |= (1 << i)
    return fingerprint


# ======================================================================
# 4. KARŞILAŞTIRMA — Hamming distance -> benzerlik (0.0 - 1.0)
# ======================================================================
def hamming_similarity(a: int, b: int, hash_bits: int = HASH_BITS) -> float:
    distance = bin(a ^ b).count('1')
    return max(0.0, 1.0 - distance / (hash_bits / 2))


# 5. SAMPLING — özellik listesini parçalara (chunk) böl
# ======================================================================
def chunk_features(
    features: list[str],
    chunk_size: int = 150,
    overlap: int = 120,
) -> list[list[str]]:
    """Shingle listesini örtüşmeli parçalara böler."""
    if not features:
        return []
    if len(features) <= chunk_size:
        return [features]

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(features):
        chunk = features[start:start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(features):
            break
        start += step
    return chunks


# 6. ANA KARŞILAŞTIRMA — chunk'ları eşleştir, ortalamasını al
# ======================================================================
def _greedy_match_similarity(hashes_a: list[int], hashes_b: list[int]) -> float:
    if not hashes_a or not hashes_b:
        return 0.0

    pairs = sorted(
        ((hamming_similarity(ha, hb), i, j)
         for i, ha in enumerate(hashes_a)
         for j, hb in enumerate(hashes_b)),
        reverse=True,
    )

    used_a, used_b = set(), set()
    total, count = 0.0, 0
    for sim, i, j in pairs:
        if i not in used_a and j not in used_b:
            total += sim
            count += 1
            used_a.add(i)
            used_b.add(j)
            if len(used_a) == len(hashes_a) or len(used_b) == len(hashes_b):
                break

    return total / count if count else 0.0


def compare_documents(
    text_a: str,
    text_b: str,
    chunk_size: int = 150,
    overlap: int = 120,
    ngram: int = NGRAM,
) -> dict:
    """İki dokümanı shingle + chunk-bazlı sampling ile karşılaştırır."""
    # tokenize -> shingle  (tokenizer korunuyor, üstüne n-gram ekleniyor)
    feats_a = make_shingles(tokenize(text_a), ngram)
    feats_b = make_shingles(tokenize(text_b), ngram)

    chunks_a = chunk_features(feats_a, chunk_size, overlap)
    chunks_b = chunk_features(feats_b, chunk_size, overlap)

    hashes_a = [simhash(c) for c in chunks_a]
    hashes_b = [simhash(c) for c in chunks_b]

    similarity = _greedy_match_similarity(hashes_a, hashes_b)

    return {
        "similarity": similarity,
        "features_a": len(feats_a),
        "features_b": len(feats_b),
        "chunks_a": len(chunks_a),
        "chunks_b": len(chunks_b),
    }


# 7. KOMUT SATIRI ARAYÜZÜ
# ======================================================================
def _print_report(result: dict) -> None:
    print("=" * 55)
    print("DOKÜMAN BENZERLİK RAPORU")
    print("=" * 55)
    print(f"  Doküman A : {result['features_a']:>7} shingle, {result['chunks_a']:>4} parça")
    print(f"  Doküman B : {result['features_b']:>7} shingle, {result['chunks_b']:>4} parça")
    print("-" * 55)
    print(f"  Benzerlik : {result['similarity']:.4f}"
          f"  (%{result['similarity']*100:.1f})")
    print("=" * 55)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) == 3:
        name_a, name_b = sys.argv[1], sys.argv[2]
    else:
        name_a, name_b = "excelFormattedDocsTesting__PortalAdmin_Uploads_Content_FastAccess_63bbb3a921784.xlsx", "excelFormattedDocsTesting__PortalAdmin_Uploads_Content_FastAccess_76ec4bd052066.xlsx"

    path_a = name_a if os.path.isabs(name_a) else os.path.join(script_dir, name_a)
    path_b = name_b if os.path.isabs(name_b) else os.path.join(script_dir, name_b)

    print(f">> Karşılaştırılıyor:\n   A = {path_a}\n   B = {path_b}\n")

    # --- AŞAMA 1: Metin çıkarma ---
    t0 = time.perf_counter()
    r1 = extract_text(path_a)
    r2 = extract_text(path_b)
    t_extract = time.perf_counter() - t0

    if not r1["success"]:
        print(f"HATA — {name_a}: {r1['error']}")
        return
    if not r2["success"]:
        print(f"HATA — {name_b}: {r2['error']}")
        return

    print(f"   {name_a}: {r1['unit_count']} {r1['unit_label']} okundu")
    print(f"   {name_b}: {r2['unit_count']} {r2['unit_label']} okundu\n")

    # --- AŞAMA 2: Karşılaştırma ---
    t0 = time.perf_counter()
    result = compare_documents(r1["text"], r2["text"])
    t_compare = time.perf_counter() - t0

    _print_report(result)

    print(f"\n  Dosya okuma süresi  : {t_extract:.3f} sn")
    print(f"  Karşılaştırma süresi: {t_compare:.3f} sn")
    print(f"  Toplam              : {t_extract + t_compare:.3f} sn")


if __name__ == "__main__":
    main()