// Array of pre-stored words for autocomplete
const wordList = [
    "politica",
    "internacional",
    "nacional", 
    "deporte",
    "ocio",
    "turismo",
    "tecnologia",
    "ciencia",
    "gastronomia",
    "viajes",
    "economia",
    "mercado",
    "vivienda",
    "corazon",
    "moda",
    "fashion",
    "educacion",
    "futbol",
    "f1",
    "baloncesto",
    "tenis",
    "golf",
    "rugby",
    "balonmano",
    "voleibol",
    "sociedad",
    "salud",
    "nutricion",
    "bienestar"
];
lastTopic = "";

// Get references to input and list elements
const input = document.getElementById("search-input");
const autocompleteList = document.getElementById("autocomplete-list");

// Event listener for input change when input event
input.addEventListener("input", () => {
    const inputValue = input.value.trim().toLowerCase();
    if (inputValue != "") {
        const matchingWords = wordList.filter(word => word.toLowerCase().includes(inputValue));
        showTopicList(matchingWords);
    }
});

// Event listener for input change when 'click' inside or outside event
document.addEventListener("click", event => {
    if (event.target.className === "input-topics") {
        if (autocompleteList.innerHTML === "" && input.value === "") {
            showTopicList(wordList);
        } else if (autocompleteList.innerHTML === "" && input.value != "") {
            const inputValue = input.value.trim().toLowerCase();
            if (inputValue != "") {
                const matchingWords = wordList.filter(word => word.toLowerCase().includes(inputValue));
                showTopicList(matchingWords);
            }
        } else if (autocompleteList.innerHTML != "" && input.value !== "") {
            autocompleteList.innerHTML = "";
        }
    } else if (autocompleteList.innerHTML != "") {
        autocompleteList.innerHTML = "";
    }
});

function showTopicList(words) {
    autocompleteList.innerHTML = "";
    words.forEach(word => {
        const listItem = document.createElement("li");
        listItem.textContent = word;
        listItem.addEventListener("click", () => {
            input.value = word;
            autocompleteList.innerHTML = "";
        });
        autocompleteList.appendChild(listItem);
    });
};

// Event listener for Enter key press
input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        // Call your templated function here with searchText
        console.log("Search text:", input.value);
        //call_django_view(input.value)
        event.preventDefault();
        window.location.href = `/${input.value}`
    }
});