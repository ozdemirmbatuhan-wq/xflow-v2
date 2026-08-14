# AeroOpt 0.8 — flow5 tabanlı çok amaçlı profil ve kanat optimizasyonu

AeroOpt, Eppler E818 veya kullanıcının verdiği bir DAT profiliyle başlar; serbest biçimli CST/Kulfan profilini ve üç istasyonlu planar kanadı gerçek flow5 analizleriyle optimize eder. İstenirse bağlı profil–kanat döngüsü bittikten sonra sabit bir NACA 4 haneli kesitle bağımsız winglet karşılaştırması çalıştırır. Varsayılan `flow5_native` zincirinde AeroOpt'un eski aerodinamik korelasyonları amaç fonksiyonuna girmez.

## Tasarım zinciri

1. E818 veya özel DAT geometri olarak yüklenir ve CST5/CST6 eğrisine uydurulur.
2. Profil adayları Differential Evolution ile oluşturulur. RBF surrogate yeterli gerçek örnek oluşunca yalnızca hangi DE önerisinin çözücüye gideceğini seçer; amaç değeri hiçbir zaman tahminle değiştirilmez. Her solver DAT'ı başlık hariç tam **100 koordinat noktası** içerir.
3. Her aday, hız aralığındaki Reynolds/Mach noktalarında flow5'in gömülü `XFoilTask` çözücüsüyle analiz edilir.
4. Profil ve kanat varsayılan iki bağlı iterasyonda çalışır: ilk kanattan alınan gerçek MAC, Reynolds ve spanwise yerel Cl hedefleri yeniden profil optimizasyonuna beslenir.
5. Kanat açıklığı, kök chord, taper, çeyrek-chord sweep, uç twist; istenirse orta-istasyon chord ve twist değişkenleri optimize edilir. Winglet bu döngüye girmez. Seçenek açıksa final planar kanat bulunduktan sonra yalnız bir winglet aşaması çalışır; aynı izdüşümsel span/hedef taşımada yükseklik–cant–toe–taper aranır ve wingletli/wingletsiz sonuçlar doğrudan karşılaştırılır.
6. Arama ağı ve final ağı sonrasında daha ince üçüncü ağla CD/hedef-Cl alfa yakınsaması kontrol edilir. Yakınsamayan, `out_of_mesh` olan veya viskoz çözümü başarısız noktalar uygun kabul edilmez.
7. İstenirse kök–orta–uç profilleri yerel Reynolds/Cl koşullarında ayrı ayrı optimize edilir ve üç profilli kanat yeniden çözülür.
8. Varsayılan kanat optimizeri gerçek NSGA-II'dir. Sürükleme ve stall kullanımı; yapısal denetim açıksa ayrıca kütle/yapısal kullanım, Pareto rütbesi ve crowding-distance ile birlikte optimize edilir. Kök eğilme momenti yalnız telemetridir; amaç, kısıt veya fizibilite koşulu değildir.
9. Profil ve kanat başlangıç bütçeleri, amaç/Pareto hareketi tolerans dışındaysa örneğin `48 → 96 → 192` biçiminde otomatik büyür. Yakınsama sağlanırsa kullanılmayan bütçe harcanmaz; azami bütçede hareket sürerse sonuç bütçe-sınırlı işaretlenir.
10. Su taraması açıksa arama adaylarında hızlı `Cp_min` kısıtı kullanılır; seçilen yüksek çözünürlüklü finalistte gerçek flow5 panel köşeleri ve Cp değerleri ayrıca alınarak 3B risk haritası, riskli alan, açıklık dağılımı ve hız/derinlik duyarlılığı hesaplanır.
11. Solver kimliği, 100-nokta sözleşmesi, kuvvet/CD kapanışı, mesh, telemetri ve fiziksel makullük kontrolleri otomatik doğrulama raporuna yazılır.
12. Sonuçlar DAT, XML, OBJ, kapalı-katı STEP, CSV, doğrulama/Pareto/teşhis/kavitasyon JSON'ları, ZIP ve yalnızca flow5 gerçekten kaydettiyse `.fl5` olarak dışa aktarılır. Fizibilite önceliği olmadan skaler puanın seçeceği kanat farklıysa onun OBJ/STEP/CSV/FL5 dosyaları da ayrıca verilir.

Uzun işlemler arka planda yürür. Arayüz gerçek aşama/aday/seed ilerlemesini gösterir ve işi iptal edebilir. İki ayrı devam katmanı vardır: SHA-256 değerlendirme önbelleği tamamlanmış flow5 yanıtlarını, optimizer checkpoint'i ise DE veya NSGA-II popülasyonunu, nesli, değerlendirme geçmişini, bütçe denetleyicisini, RNG durumunu ve varsa surrogate örneklerini atomik JSON olarak saklar. Checkpoint nesil sınırlarında alınır; neslin ortasında iptal edilirse son tamamlanmış nesilden devam edilir. Başarıyla biten problem checkpoint'i temizlenir.

## 0.8 optimizasyon ve karar araçları

| Özellik | Varsayılan | Davranış |
|---|---:|---|
| Kanat NSGA-II | Açık | Fizibiliteyi önceleyen non-dominated sorting, crowding-distance, ikili turnuva, SBX ve sınırlı mutasyonla Pareto cephesini arama sırasında üretir |
| Otomatik bütçe | Açık | Profilde en iyi skaler amacı; kanatta skaler uzlaşma ve Pareto ideal noktasını izler, tolerans dışındaysa bütçeyi 2× büyütür; varsayılan üst sınır başlangıcın 4×'idir |
| Surrogate ön eleme | Açık | Her gerçek çözüm için varsayılan 6 DE önerisini RBF ile sıralar; seçilen aday ve tüm finalistler gerçek flow5 ile çözülür |
| Optimizer checkpoint | Açık | Popülasyon + RNG + surrogate durumunu problem parmak iziyle saklar; aynı ayarlarla yeniden başlatınca geri yükler |
| Multi-seed | 1 koşu | Arayüzden 3 veya 5 bağımsız seed seçilebilir; en iyi fizibil koşu seçilir, amaç/geometri CV raporlanır |
| Pareto analizi | Açık | NSGA-II seçilince cephe optimizer tarafından üretilir; DE/adaptive seçilince gerçek çözücü adaylarından sonradan çıkarılır |
| Winglet tasarımı | Kapalı | Bağlı döngü ve spanwise profil iyileştirmesi bittikten sonra seçilen planar geometri dondurulur. Varsayılan NACA 0012 veya girilen NACA 4 haneli kesitle yalnız bir son aşamada yükseklik, cant, toe ve taper aranır; wingletli/wingletsiz sonuçlar birlikte raporlanır |
| Doğrulama/regresyon | Açık | Dokuz tutarlılık/makullük kontrolü ve tekrarlanabilir SHA-256 sonuç imzası üretir |
| Otomatik teşhis | Açık | Stall, Reynolds, drag bileşeni, mesh, solver noktası, sınır, coupling, seed, yapı ve kavitasyon kanıtlarını kurallarla sıralar |
| Proje geçmişi | Açık | Son 12 tasarım özetini tarayıcı yerel depolamasında tutar ve iki tasarımı yan yana karşılaştırır |

Surrogate, CFD/flow5 yerine geçen bir fizik modeli değildir. Yalnızca pahalı gerçek değerlendirmeye gönderilecek DE önerisini seçerek aramayı yönlendirir. Kanat NSGA-II modunda skaler surrogate, Pareto çeşitliliğini bozmaması için otomatik devre dışı bırakılır; bütün NSGA-II adayları doğrudan flow5 ile puanlanır. Doğrulama raporu deneysel doğrulama veya RANS karşılaştırması değil; solver sözleşmesi, sayısal kapanış ve analitik makullük denetimidir.

## İsteğe bağlı mühendislik kontrolleri

| Kontrol | Varsayılan | Ne yapar | Sınırı |
|---|---:|---|---|
| Yapısal denetim | **Kapalı** | flow5 spanwise yüküyle Euler–Bernoulli eğilme, kapalı ince cidarlı burulma, gerilme/sehim/twist kullanımı ve yaklaşık malzeme kütlesi | FEA değildir; deforme kanadı tekrar aerodinamik olarak çözmez |
| Su/kavitasyon | Su seçilince kullanılabilir | Aramada flow5 `Cp_min`; finalistte panel Cp/alan/geometrisiyle 3B risk haritası, emniyetli–yakın–riskli alan, fiziksel başlangıç alanı, basınç açığı, şiddet ve hız/derinlik duyarlılığı | Tek fazlı başlangıç taramasıdır; kavite boyu/hacmi veya L/D kaybı değildir |

Yapısal kontrolü istemiyorsanız arayüzde **Yapısal denetimi etkinleştir** kutusunu boş bırakın. Bu durumda yapısal hesap yapılmaz ve optimizasyon puanına hiçbir yapısal ceza eklenmez. Kutuyu işaretlerseniz yalnızca ön boyutlandırma amaçlı sonuçlar amaç fonksiyonuna ve uygunluk kararına katılır.

Kavitasyon için **Sert kısıt** seçimi `Cp_min` kullanımını aday uygunluğuna ve amaç fonksiyonuna dahil eder. **Yalnız raporla** seçimi aynı finalist panel haritasını ve etki göstergelerini üretir fakat L/D tabanlı tasarım seçimini değiştirmez. Renkli harita seçilen yüksek çözünürlüklü final çözümünden gelir; binlerce arama adayında panel geometrisi taşınmadığı için arama maliyeti gereksiz yere büyümez.

## Windows'ta en kolay kullanım

Depoyu GitHub'a yükledikten sonra yerel bilgisayara Python, Qt, Visual Studio, flow5 veya Gmsh kurmadan paket oluşturabilirsiniz:

1. GitHub'da **Actions** sekmesini açın.
2. **Build Windows EXE with flow5 7.57** iş akışında **Run workflow** seçin.
3. İş bitince `AeroOpt-0.8.0-Windows-flow5` artifact'ini indirin.
4. ZIP'i tamamen çıkarıp `AeroOpt.exe` dosyasını çalıştırın.

İş akışı Python testlerini çalıştırır; sabitlenmiş flow5 kaynaklarını ve C++ runner'ı derler; gerçek E818/100-nokta profil, VLM2, TRIUNIFORM winglet, panel Cp haritası, telemetri, `.fl5` ve OpenCascade STEP-katısı smoke testlerini geçmeden paketi yayımlamaz.

Paket içindeki uygulamada **flow5 runner yolu** boş bırakılır. Arayüz `aeropt-flow5-runner.exe` dosyasını otomatik bulur. `flow5.exe`, `flow5-runner.exe` veya `fake_flow5_runner.exe` bu alan için uygun değildir.

## Kaynaktan çalıştırma

Python 3.10+ gerekir; Windows için 64-bit Python 3.12 önerilir.

Windows:

1. Python 3.12 Windows 64-bit kurucusunu indirin ve **Add python.exe to PATH** seçeneğini işaretleyin.
2. Kaynak ZIP'ini tamamen çıkarın.
3. `start_windows.bat` dosyasına çift tıklayın.
4. Tarayıcı açılmazsa `http://127.0.0.1:8765` adresini açın.

Linux/macOS:

```bash
chmod +x start_linux_macos.sh
./start_linux_macos.sh
```

Manuel başlatma:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python server.py
```

Kaynak sürümde ayrıca flow5 runner gerekir. `flow5.exe` masaüstü uygulamasının genel amaçlı optimizasyon CLI'ı olmadığı için köprü, flow5 C++ API'sindeki `XFoilTask`, `PlaneTask`, `PanelAnalysis`, çalışma noktaları ve proje kaydetme çağrılarını kullanır.

## Uyumlu flow5/XFOIL sürümü

Runner tam olarak **flow5 v7.57** API'sine ve GitHub iş akışında şu kaynak commit'ine sabitlenmiştir:

```text
a9e852c559590188e00e9efe997c35c1dec7209b
```

Runner yanıtında sürüm `7.57` değilse Python adaptörü işlemi reddeder. Daha yeni flow5 API sürümleri kaynak uyumluluğu doğrulanmadan desteklenmiş sayılmaz. Harici XFOIL 6.99 gerekmez; 2B profil analizi flow5'in içindeki XFoil ile yapılır. XFLR5 kullanılmaz.

Runner'ı elle derlemek için Visual Studio 2022 C++, CMake 3.20+, Qt 6, Gmsh, OpenCascade ve flow5 ile aynı lineer cebir/runtime bağımlılıkları gerekir. Ayrıntılar `flow5_bridge/README.md` içindedir.

## Arayüzden önerilen ilk çalışma

1. Akışkanı, referans/minimum/maksimum hızı, hız noktası sayısını ve toplam hedef taşımayı girin.
2. Kanat sınırlarını girin; E818, CST6 ve 100 profil noktası seçili kalabilir.
3. Hızlı kurulum kontrolünde profil/kanat başlangıç bütçelerini `8 / 8`, finalist sayısını `1` yapın. Mutlak kısa smoke test istiyorsanız otomatik bütçe denetimini geçici olarak kapatın; açıkken arama gerek görürse `8 → 16 → 32` büyür.
4. Mesh yakınsamasını açık bırakın. Arama/final/ince varsayılanları sırasıyla `10×14`, `14×22`, `20×32`'dir.
5. Yapısal hesabı istemiyorsanız yapısal anahtarı kapalı bırakın. Su seçtiyseniz derinlik, buhar basıncı, kavitasyon güvenlik katsayısı ve sert-kısıt/yalnız-rapor politikasını kontrol edin.
6. İlk kontrol için multi-seed değerini `1` bırakın; surrogate, checkpoint ve otomatik doğrulamayı açık bırakın.
7. **Optimizasyonu başlat** düğmesine basın. Üretim ön tasarımında `48 / 48 / 3`, otomatik bütçe `2× / 4× / %3` ve doğrulama koşusu için multi-seed `3` iyi bir başlangıçtır.

Kök–orta–uç profillerini ayrı optimize etme seçeneği varsayılan olarak kapalıdır; açıldığında iki ek profil optimizasyonu ve bir üç-profilli final kanat çözümü yaptığı için süreyi belirgin artırır.

Tek bir flow5 adayının zaman aşımı için üst sınır yoktur; arayüze en az 30 saniye olmak üzere ihtiyacınız kadar büyük bir değer girebilirsiniz. Bu ayar bütün optimizasyonun toplam süresi değil, her ayrı runner çağrısının üst sınırıdır.

## Optimizasyon ve çözücü ayrıntıları

### Profil

- NACA numarasıyla sınırlı değildir; CST üst/alt yüzey katsayıları sürekli eğriyi tanımlar.
- Varsayılan optimizer Differential Evolution'dır; eski `adaptive_elite` karşılaştırma seçeneği korunmuştur.
- Surrogate en az değişken sayısı + 2 benzersiz gerçek örnekten sonra eğitilir. Holdout hatası kullanıcı sınırını geçiyorsa erken durdurma yapamaz.
- Otomatik bütçe açıksa durma kararını surrogate değil bütçe denetleyicisi verir. Otomatik bütçe kapatıldığında profil DE aramasındaki `solver_evaluations_saved`, yalnız güvenilir surrogate hatası ve iyileşme eşiğiyle erken durdurulan gerçek sabit bütçeyi ifade eder; ön elenen sanal öneriler gerçek değerlendirme sayılmaz.
- Amaç tek hızdaki tepe L/D değil, hız aralığında hedef Cl'lerde ortalama ve kötü-durum drag ile stall/polar sınırı cezalarının bileşimidir.
- Finalistler daha ince alfa aralığında tekrar analiz edilir. Kazanç eşikten küçükse baseline korunur.

### Bağlı profil–kanat döngüsü

- Tasarım Cl değeri kullanıcı tarafından verilmişse döngü boyunca kilitli kalır.
- Kanat çözümünden MAC ve spanwise yük programı çıkarılır; sonraki profil iterasyonunun Reynolds ve Cl çalışma noktaları güncellenir.
- En iyi uygun profil–kanat çifti seçilir. Cl programı ve amaç değişimi toleranslara gelmezse sonuç `review` olarak işaretlenir.

### Kanat

- Üç geometri istasyonu vardır: kök, yarı-açıklığın ortası ve uç.
- Orta chord/twist bağımsız değişkenleri kapatılırsa orta istasyon kök–uç arasında doğrusal enterpolasyon olur.
- Arama varsayılan VLM2, finalist çözümü TRIUNIFORM'dur.
- Varsayılan optimizer NSGA-II'dir. Fizibil olmayan adaylar toplam kısıt ihlaline göre; fizibil adaylar sürükleme–stall ve etkinleştirilen mühendislik amaçlarının Pareto rütbesi/crowding mesafesine göre seçilir. Kök eğilme momenti bu sıralamaların hiçbirine girmez.
- Final doğrulamasına fizibilite-öncelikli uzlaşma adayı ile Pareto cephesinin seyrek bölgelerinden temsilciler gönderilir. Final teslimi önce en düşük sert-kısıt ihlaline, eşitlikte en düşük skaler toplam amaca göre seçilir.
- Aynı arama havuzundaki en düşük skaler amaçlı aday finalist kotasını tüketmeden ayrıca korunur. Fizibilite önceliği kullanılmadan seçilecek bu kanat final ağında yeniden çözülür; ana kanattan farklıysa performans/geometri karşılaştırması ile ayrı OBJ, kapalı-katı STEP, CSV ve `.fl5` dosyaları üretilir.
- Viskoz profil drag'i gömülü XFoil'den; 3B/indüklenmiş bileşen ve spanwise dağılım flow5 çalışma noktalarından gelir.
- İnce ağdaki sonuç final `.fl5` projesine ve metre birimli OpenCascade STEP katısına yazılır. Eş alanlı dikdörtgen baseline aynı final yöntem/ağ ile karşılaştırılır.
- Winglet seçeneği kapalıysa winglet çözücüsü hiç çağrılmaz. Açıksa profil–kanat bağlı döngüsü ve isteğe bağlı kök–orta–uç profil iyileştirmesi tamamen wingletsiz biter. Son planar geometri dondurulur; toplam izdüşümsel span korunarak ana kanat yarı-açıklığı winglet yatay izdüşümü kadar kısaltılır ve dördüncü yüksek-dihedral kesitte sabit `winglet_naca_code` profiliyle yalnız yükseklik, cant, toe ve taper optimize edilir. Bu son aşama döngüye geri beslenmez. Son karar fizibiliteyi önceleyip aynı planform/hedef taşıma koşulundaki wingletli ve wingletsiz sonuçları karşılaştırır.

### Kavitasyon haritası ve etki göstergeleri

- Arama boyunca her adayın en kötü flow5 `Cp_min` değeri Bernoulli hidrostatik basıncı ve kullanıcı emniyet katsayısıyla hızlıca taranır.
- Yalnız seçilen yüksek çözünürlüklü finalist için hedef taşımaya en yakın gerçek `PlaneOpp` noktasındaki panel köşeleri, alanı, yüzey normali ve Cp alınır. TRIUNIFORM/TRILINEAR için üç düğüm Cp'sinin panel ortalaması; QUADS/VLM için `WingOpp` panel Cp'si kullanılır.
- Panel kullanımı `U = emniyet × max(-Cp, 0) / σ`, `σ = (p_atm + ρgh - p_v) / q` olarak hesaplanır. Haritada `U < 0,8` güvenli, `0,8 ≤ U < 1` sınıra yakın ve `U ≥ 1` riskli gösterilir.
- Rapor; riskli alan yüzdesi, emniyet katsayısız fiziksel başlangıç alanı, maksimum kullanım, alan-ağırlıklı şiddet, buhar basıncı altındaki basınç açığı–alan integrali, bileşen/yüzey ve açıklık dağılımlarını içerir.
- Hız ve derinlik grafikleri finalist Cp alanını sabit tutan beşer noktalı duyarlılık taramasıdır. Bunlar çok fazlı yeniden çözüm değildir; gerçek kavite boyu/hacmi, ventilasyon veya kavitasyon kaynaklı drag/L/D kaybı iddia edilmez.

### 16 çekirdek

Profil aşamasında Reynolds noktaları ve adaylar, toplam eşzamanlı bütçe yaklaşık `flow5_threads` olacak biçimde dış süreçlere dağıtılır. Kanat adayları sırayla çalışır; flow5 panel çözümüne en çok 16 iç thread verilir. Böylece `16 süreç × 16 thread` aşırı aboneliği oluşturulmaz. Gerçek ölçeklenme mesh ve flow5'in derleme bağımlılıklarına bağlıdır.

### Otomatik bütçe, multi-seed ve Pareto

- Girilen foil/kanat bütçeleri artık **başlangıç bütçesidir**. Varsayılan `growth=2`, `maximum multiplier=4` ile 48 adaydan başlayan aşama en çok 192 gerçek çözüme gider.
- İlk kontrolde bütçenin ilk yarısı ile tamamı; sonraki kontrollerde ardışık bütçe kilometre taşları karşılaştırılır. Profilde en iyi amaç değişimi, kanatta buna ek olarak Pareto ideal noktasının en büyük göreli hareketi kullanılır.
- Son değişim toleransın altındaysa `converged`; azami bütçede üstündeyse `budget_exhausted` üretilir ve genel sonuç `review` olur. Bu, aerodinamik çözümün başarısız olduğu değil optimizer yeterliliğinin henüz kanıtlanmadığı anlamına gelir.

- `1` seed normal optimizasyondur; `3` ve `5` seçenekleri tam zinciri bağımsız tohumlarla tekrarlar. Süre yaklaşık koşu sayısıyla çarpılır, ancak ortak flow5 değerlendirmeleri cache'den gelebilir.
- Seçim önce fizibiliteye, sonra skaler amaç değerine göre yapılır. Amaç CV veya geometri CV kullanıcı toleransını aşarsa sonuç `review` olur.
- NSGA-II Pareto cephesi optimizerın seçim baskısıyla doğrudan oluşur. DE/adaptive modunda aynı grafik sonradan çıkarılan bir karşılaştırmadır; arayüz bu köken farkını ve arama/final mesh fidelity farkını açıkça gösterir.

## Sonuçları yorumlama

- `2B L/D`, gömülü XFoil profil polarından gelir.
- `3B L/D`, profil ve indüklenmiş drag dahil kanat sonucudur.
- Mesh yakınsaması geçmediyse, viskoz istasyonlar yakınsamadıysa veya bir çalışma noktası `out_of_mesh` ise tasarım uygun sayılmaz.
- Düşük Reynolds, yüksek hedef Cl, dar geometri zarfı veya stall kenarı düşük L/D verebilir; AeroOpt flow5 sonucunu kendi korelasyonuyla yükseltmez.
- Yapısal ve kavitasyon kutuları yalnızca ön taramadır. Kritik tasarım kararları deney, FEA ve uygun yüksek doğrulukta CFD ile doğrulanmalıdır.
- Doğrulama panelinde bloklayan bir satır başarısızsa sonuç `review` olur. Uyarı niteliğindeki lift-eğimi/span-verimi kontrolleri raporlanır fakat tek başına fizibiliteyi bozmaz.
- Otomatik teşhis deterministik kanıt kurallarıdır; kullanıcı girdilerini kendiliğinden değiştirmez. Önerilen sınır değişikliğini yeni bir koşuda siz uygularsınız.

## Python API

```python
from aeropt import run_design

result = run_design({
    "flow": {
        "speed_m_s": 18,
        "speed_min_m_s": 14,
        "speed_max_m_s": 22,
        "speed_samples": 5,
        "target_lift_n": 120,
    },
    "airfoil": {
        "baseline_profile": "e818",
        "cst_order": 6,
        "solver_coordinate_points": 100,
    },
    "wing": {
        "winglet_optimization_enabled": True,
        "winglet_naca_code": "0012",
        "winglet_height_min_m": 0.10,
        "winglet_height_max_m": 0.35,
        "winglet_cant_min_deg": 65.0,
        "winglet_cant_max_deg": 90.0,
        "winglet_toe_min_deg": -3.0,
        "winglet_toe_max_deg": 3.0,
        "winglet_taper_min": 0.35,
        "winglet_taper_max": 0.85
    },
    "solver": {
        "airfoil_strategy": "flow5_native",
        "flow5_runner_path": r"C:\Tools\aeropt-flow5-runner.exe",
        "flow5_threads": 16,
        "flow5_foil_candidate_budget": 48,
        "flow5_wing_candidate_budget": 48,
        "flow5_winglet_candidate_budget": 48,
        "flow5_timeout_seconds": 21600,
        "flow5_wing_optimizer": "nsga2",
        "flow5_budget_escalation_enabled": True,
        "flow5_budget_growth_factor": 2.0,
        "flow5_budget_maximum_multiplier": 4.0,
        "flow5_budget_convergence_tolerance_percent": 3.0,
        "flow5_coupled_iterations": 2,
        "flow5_mesh_convergence_enabled": True,
        "flow5_surrogate_enabled": True,
        "flow5_surrogate_proposals_per_evaluation": 6,
        "flow5_checkpoint_enabled": True,
        "flow5_multi_seed_runs": 3,
    },
    "hydro": {
        "enabled": True,
        "constraint_mode": "report_only",
        "submergence_depth_m": 1.0,
        "cavitation_safety_factor": 1.2
    },
    "structure": {"enabled": False},
    "validation": {"enabled": True},
})

print(result["airfoil"])
print(result["wing"]["geometry"], result["wing"]["ld"])
```

Eksiksiz giriş örneği `example_request.json` dosyasındadır.

## Test

```bash
python -m unittest discover -s tests -v
node --check static/app.js
```

Python testleri orkestrasyonu deterministik bir protokol ikiziyle doğrular. Bu ikiz aerodinamik çözücü değildir ve üretim arayüzünde `AEROPT_ALLOW_TEST_DOUBLE=1` olmadan reddedilir. Gerçek flow5 C++ derlemesi GitHub Windows iş akışındaki ayrı smoke testiyle doğrulanır.

## Proje yapısı

```text
aeropt/
  flow5.py               # güvenli JSON/subprocess adaptörü, cache ve iptal
  flow5_optimization.py  # DE profil + NSGA-II planform, Pareto, mesh ve telemetri
  convergence.py         # otomatik bütçe kilometre taşları ve yeterlilik kararı
  flow5_pipeline.py      # bağlı döngü, üç profil ve export orkestrasyonu
  surrogate.py           # gerçek flow5 adaylarını seçen RBF ön eleme
  checkpoint.py          # atomik DE/NSGA-II popülasyon, RNG ve bütçe checkpoint'i
  validation.py          # solver/regresyon ve fiziksel makullük sözleşmesi
  diagnostics.py         # kanıta dayalı otomatik başarısızlık teşhisi
  reliability.py         # multi-seed seçim/kararlılık ve rapor export'u
  structures.py          # isteğe bağlı ön yapısal tarama
  hydro.py               # su için kavitasyon/serbest-yüzey ön taraması
  baselines.py           # E818 ve türetilmiş bağlı-iterasyon baseline'ları
  airfoil.py             # DAT, tam 100 nokta ve CST geometrisi
  exporters.py           # DAT/XML/OBJ/CSV/JSON/ZIP
flow5_bridge/             # flow5 7.57 C++ API runner
static/                   # yerel HTML/CSS/JS arayüz
ci/                       # gerçek flow5 build/runtime/smoke betikleri
.github/workflows/        # tek tıklamalı Windows EXE üretimi
tests/                    # geometri, protokol, Pareto, surrogate, checkpoint ve mühendislik testleri
server.py                 # localhost sunucusu ve arka-plan iş API'si
```

## Kapsam dışı kalanlar

Bu sürüm, yol haritasındaki gerçek Windows flow5 derlemesini bu çalışma ortamında gerçekleştirmez; bunu GitHub Actions iş akışı yapar. Üretim/imalat geometrisi kısıtları da henüz eklenmemiştir. Tam RANS/LES, çok fazlı serbest yüzey, geçiş modeli doğrulaması, flutter, divergence veya tam FEA çözümü yoktur.

## Lisans

Python web uygulaması MIT lisanslıdır. flow5 kütüphanelerine bağlanan `flow5_bridge` GPL-3.0-or-later lisanslıdır. Kaynak ZIP'i üçüncü taraf ikililerini içermez; GitHub Actions artifact'i gereken runtime dosyaları ve lisans bildirimleriyle paketlenir. Yeniden dağıtımdan önce `THIRD_PARTY_NOTICES.md` ve upstream lisanslarını inceleyin.
