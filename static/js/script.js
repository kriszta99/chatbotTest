const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const chatbox = document.querySelector(".chatbox");
const chatbotToggler = document.querySelector(".chatbot-toggler");
const chatbotCloseBtn = document.querySelector(".close-btn");

let userMessage=null;
const inputInitHeight = chatInput.scrollHeight;

const generateLinkHTML = (url) => {
    return `<a href="${url}" target="_blank" rel="noopener">${"\nKattints ide\n"}</a>`;
};

const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    const chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">robot_2</span><p></p>`;
    chatLi.innerHTML = chatContent;

   let formattedMessage = message.replace(
    /[(\[\s]*https?:\/\/[^\s\[\]\(\),;]+[)\],;]*/g,
    (match) => {
        const cleanUrlMatch = match.match(/https?:\/\/[^\s\[\]\(\),;]+/);
        if (!cleanUrlMatch) return match;
        const cleanUrl = cleanUrlMatch[0];
        return generateLinkHTML(cleanUrl);
    }
    );

    if (/^\d+\.\s/m.test(formattedMessage)) {
    // Számozott lista
    const lines = formattedMessage.split("\n");
    const listItems = lines.map(line => {
        const match = line.match(/^\d+\.\s(.*)/);
        return match ? `<li>${match[1]}</li>` : line;
    });
    formattedMessage = "<ol>" + listItems.join("") + "</ol>";
    } else if (/^[*-]\s/m.test(formattedMessage)) {
        // Csillagos vagy kötőjeles lista
        const lines = formattedMessage.split("\n");
        const listItems = lines.map(line => {
        const match = line.match(/^[*-]\s{1,2}(.*)/);
        return match ? `<li>${match[1]}</li>` : line;
        });
        formattedMessage = "<ul>" + listItems.join("") + "</ul>";
    } else {
        formattedMessage = formattedMessage.replace(/\n/g, "<br>");
    }


    chatLi.querySelector("p").innerHTML = formattedMessage;
    return chatLi;
};

const handleChat = async () => {
    //a felhasználó által beírt üzenet, és távolítom el a felesleges szóközöket
    userMessage = chatInput.value.trim();
    if(!userMessage) return;
    //megtisztítom az input szövegmezőt, és  visszaállitom az alapértelmezett magasságot
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;

    //a chatbox-hoz hozzáadom a felhasznaló üzenetét 
    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);

    /*setTimeout(() => {
        // Display "Thinking..." message while waiting for the response
        const incomingChatLi = createChatLi("Thinking...", "incoming");
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);
    },600)
    */
    const incomingChatLi = createChatLi("Gondolkodom...", "incoming");
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);
    try {
        const response = await fetch('/chatbot', {  //ide küldjük a kérést
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: userMessage })
        });
    
        const data = await response.json();
        console.log("Válasz:", data.answer);  //kiírjuk a választ a konzolra (debugging célból)
    
        //az érkező választ megjelenítem a chatboxban
        chatbox.removeChild(incomingChatLi);
        chatbox.appendChild(createChatLi(data.answer, "incoming"));  //a válasz megjelenik itt
        chatbox.scrollTo(0, chatbox.scrollHeight);
    
    } catch (error) {
        console.error("Hiba történt:", error);
    }
    

}
chatInput.addEventListener("input", () => {
    // itt igazitom az input szövegmező magasságát a tartalmához
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
  });

  chatInput.addEventListener("keydown", (e) => {
    //ha az Enter billentyűt a Shift nélkül van lenyomva
    if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 400){
        e.preventDefault();
        handleChat();
    }
   
  });
sendChatBtn.addEventListener("click",handleChat);
chatbotToggler.addEventListener("click", ()=> document.body.classList.toggle("show-chatbot"));
chatbotCloseBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));