# Naïve Bayes Sınıflandırma Yöntemi İle Meme Kanseri Teşhisi

## Özet

&emsp;&emsp;Günümüzde artarak devam eden meme kanseri sorununun teşhis edilmesi aşamasında, bilime katkıda bulunması amacıyla, ABD’nin Wisconsin eyaletindeki meme kanseri hastalarına ait veriler baz alınarak, Naïve Bayes yöntemi ile bir sınıflandırıcı geliştirilmiştir. Önerilen yöntem, daha önce aynı veri seti ile yapılmış bir akademik makale ile karşılaştırılmış ve sonuçlarıyla birlikte sunulmuştur. Meme kanserinin erken teşhis edilmesi konusunda her geçen gün ilerlemeler kaydedilmektedir. Geliştirilen sınıflandırıcı %96 oranında doğruluğa sahip olmasıyla beraber erken teşhis aşamasında büyük bir rol oynaması beklenmektedir.

## Kanser ve Meme Kanseri

Araştırmalara göre dünyada her yıl 2.1 milyon kadın, ülkemizde ise 20 bin kadın meme kanserinden etkilenmektedir. Yaşam boyu her 8 kadından biri meme kanseri riski, her 38 kadından birisi ise meme kanserinden ölme riski ile karşı karşıyadır.<sub>[1]</sub> İnsan vücudu, her biri kendine özgü işlevi olan milyonlarca hücreden oluşmaktadır. Vücudumuzdaki sağlıklı hücreler bölünebilme yeteneğine sahiptirler. Yaşamın ilk yıllarında hücreler daha hızlı bölünürken, erişkin yaşlarda bu hız yavaşlar. Fakat hücrelerin bu yetenekleri sınırlıdır, sonsuz bölünemezler. Her hücrenin hayatı boyunca belli bir bölünebilme sayısı vardır. Sağlıklı bir hücre ne kadar bölüneceğini bilir ve gerektiğinde ölmesini de bilir. Normalde vücudun sağlıklı ve düzgün çalışması için hücrelerin büyümesi, bölünmesi ve daha çok hücre üretmesine gereksinim vardır. Bazen buna rağmen süreç doğru yoldan sapar. Yeni hücrelere gerek olmadan hücreler bölünmeye devam eder. Bu hücrelerin düzensiz büyümesi kansere sebep olmaktadır. Bu sayede hücreler kontrolsüz olarak bölünür ve büyürler, tümör denilen anormal doku kütlesini oluştururlar. Her bir tümör kanserli olmamasına rağmen, vücudun normal işleyişini bozan sindirim, sinir ve dolaşım sistemlerini geliştirir ve istila eder.<sub>[2]</sub> Tümörler iyi huylu veya kötü huylu olabilirler. İyi huylu tümörler kanser değildir. Bunlar sıklıkla alınırlar ve çoğu zaman tekrarlamazlar. İyi huylu tümörlerdeki hücreler vücudun diğer taraflarına yayılmazlar. En önemlisi iyi huylu tümörler nadiren hayatı tehdit ederler. Kötü huylu tümörler kanserdir. Bu tümörler normal dokuları sıkıştırabilirler, içine sızabilirler ya da tahrip edebilirler. Eğer kanser hücreleri oluştukları tümörden ayrılırsa, kan ya da lenf dolaşımı aracılığı ile vücudun diğer bölgelerine gidebilirler. Gittikleri yerlerde tümör kolonileri oluşturur ve büyümeye devam ederler. Kanserin bu şekilde vücudun diğer bölgelerine yayılması olayına metastaz adı verilir.<sub>[3]</sub> Meme kanseri meme hücrelerinde başlayan kanser türüdür. Akciğer kanserinden sonar dünyada görülme sıklığı en yüksek olan kanser türüdür. Erkeklerde de görülmekler beraber kadın vakaları erkek vakalarından 100 kat fazladır. 1970’lerden bu yana meme kanserinin görülme sıklığında artış yaşanmaktadır ve bu artışa modern, Batılı yaşam tarsi sebep olarak gösterilmektedir. Kuzey Amerika ve Avrupa ülkelerinde görülme sıklığı, dünyanın diğer bölgelerinde görülme sıklığından daha fazladır. Meme kanseri, yayılmadan önce, erken tespit edilirse, hasta %96 yaşam şansına sahiptir. Her yıl 44000’de bir kadın meme kanserinden ölmektedir. Meme kanserine karşı en iyi koruyucu yöntem erken teşhistir. Memedeki iyi huylu veya kötü huylu olduğu kesin anlamanın tek yolu vardır. Biyopsi ile mikroskobik tetkik sonucu tanı koymak. Ama bazı özellikler var ki, o kitlenin daha çok neye benzediği konusunda muayene eden hekime ortalama bir fikir verebilir.<sub>[4]</sub>

# Naïve Bayes

- BAYES TEOREMİ
    - Bayes teoremi, olasılık kuramı içinde incelenen
    önemli bir konudur. Bu teorem bir rastgele değişken için
    olasılık dağılımı içinde koşullu olasılıklar ile marjinal
    olasılıklar arasındaki ilişkiyi gösterir. Bu kavram için
    Bayes Kuralı, Bayes Savı veya Bayes Kanunu adları da
    kullanılır.
    Bayes Teoreminde B olasılığının gerçekleşme
    durumu altında A olasılığının gerçekleşme durumu
    Denklem 2.1 ile açıklanabilir.