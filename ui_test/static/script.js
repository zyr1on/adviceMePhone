const input = document.getElementById("userInput");
const message = document.getElementById("message");
const phoneResults = document.getElementById("phone-results");
const chatArea = document.getElementById("chat-area");
const sendButton = document.getElementById("composer-submit-button");
const promptContainer = document.getElementById("prompt-container");

let isFirstMessage = true;

function typeWriter(text, element, speed = 50) {
  element.textContent = "";
  let i = 0;
  function type() {
    if (i < text.length) {
      element.textContent += text.charAt(i);
      i++;
      setTimeout(type, speed);
    }
  }
  type();
}

function addUserBubble(text) {
  const bubble = document.createElement('div');
  bubble.className = 'user-bubble';
  bubble.textContent = text;
  chatArea.appendChild(bubble);
  
  // Baloncukları kalıcı yap - silinmesin
}

function getColorClass(color) {
  const colorMap = {
    'siyah': 'color-siyah',
    'beyaz': 'color-beyaz',
    'kırmızı': 'color-kırmızı',
    'mavi': 'color-mavi',
    'yeşil': 'color-yeşil',
    'altın': 'color-altın',
    'mor': 'color-mor'
  };
  return colorMap[color.toLowerCase()] || 'color-siyah';
}

function displayPhones(phones) {
  if (phones.length === 0) {
    phoneResults.innerHTML = '';
    phoneResults.classList.remove('show');
    return;
  }

  const phoneGrid = phones.map(phone => {
    const trendyolLink = phone.trendyol_link && phone.trendyol_link.trim() !== '' 
      ? `<a href="${phone.trendyol_link}" target="_blank" class="store-link trendyol-link">Trendyol</a>`
      : `<span class="store-link disabled">Trendyol</span>`;
    
    const hepsiburadaLink = phone.hepsiburada_link && phone.hepsiburada_link.trim() !== '' 
      ? `<a href="${phone.hepsiburada_link}" target="_blank" class="store-link hepsiburada-link">Hepsiburada</a>`
      : `<span class="store-link disabled">Hepsiburada</span>`;

    return `
      <div class="phone-card">
        <div class="phone-brand">${phone.brand}</div>
        <div class="phone-model">${phone.model}</div>
        <div class="phone-specs">
          <div class="spec-item">
            <span class="spec-label">Depolama</span>
            <span class="spec-value">${phone.storage}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Kamera</span>
            <span class="spec-value">${phone.camera}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Renk</span>
            <span class="spec-value">
              ${phone.color}
              <span class="phone-color ${getColorClass(phone.color)}"></span>
            </span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Yıl</span>
            <span class="spec-value">${phone.year}</span>
          </div>
        </div>
        <div class="phone-price">${phone.price.toLocaleString('tr-TR')} ₺</div>
        <div class="phone-links">
          ${trendyolLink}
          ${hepsiburadaLink}
        </div>
      </div>
    `;
  }).join('');

  phoneResults.innerHTML = `<div class="phone-grid">${phoneGrid}</div>`;
  phoneResults.classList.add('show');
}

async function handleSubmit() {
  const userText = input.value.trim();
  if (userText !== "") {
    // İlk mesajsa prompt kutusunu aşağıya kaydır
    if (isFirstMessage) {
      promptContainer.classList.add('moved');
      isFirstMessage = false;
    }
    
    // Kullanıcı metnini baloncuk olarak ekle
    addUserBubble(userText);
    
    try {
      const response = await fetch('/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: userText })
      });
      
      const data = await response.json();
      typeWriter(data.response, message, 30);
      
      // Telefon sonuçlarını göster
      if (data.phones) {
        setTimeout(() => {
          displayPhones(data.phones);
        }, 1000);
      }
      
    } catch (error) {
      typeWriter("Bir hata oluştu.", message, 50);
      phoneResults.innerHTML = '';
      phoneResults.classList.remove('show');
    }
    
    input.value = "";
  }
}

input.addEventListener("keypress", function (e) {
  if (e.key === "Enter") handleSubmit();
});

sendButton.addEventListener("click", handleSubmit);
