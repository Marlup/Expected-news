// Array of pre-stored words for autocomplete
const wordList = ["apple", "banana", "data", "balloon", "cherry", "date", "grape", "kiwi", "lemon", "orange", "pear"];

// Get references to input and list elements
const input = document.getElementById("search-input");
const autocompleteList = document.getElementById("autocomplete-list");

// Function to filter and display autocomplete suggestions
function showAutocompleteSuggestions() {
    const inputValue = input.value.toLowerCase();
    autocompleteList.innerHTML = "";

    const matchingWords = wordList.filter(word => word.toLowerCase().includes(inputValue));
    matchingWords.forEach(word => {
        console.log(word)
        const listItem = document.createElement("li");
        listItem.textContent = word;
        listItem.addEventListener("click", () => {
            input.value = word;
            autocompleteList.innerHTML = "";
        });
        autocompleteList.appendChild(listItem);
    });
}

// Event listener for input change
input.addEventListener("input", showAutocompleteSuggestions);

// Event listener for Enter key press
input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        const searchText = input.value;
        // Call your templated function here with searchText
        console.log("Search text:", searchText);
    }
});