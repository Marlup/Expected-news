// From https://chat.openai.com/c/a02a6e66-2512-417e-b7e5-7de1ed5e6ed7
//import { TEST_BASE_PATH } from './constants.js';
const TEST_BASE_PATH = 'http://localhost:8000';
// JavaScript code to implement infinite scrolling
const msgOriginalUrl = "Ir al artículo original";
let isLoading = false;
const N_LOAD_ELEM = 10; // Number of items to load per request


function loadMoreNews() {
    if (!isLoading) {
        isLoading = true;
        var offset = document.getElementsByClassName("scrollable-content").length;
        var newsContainer = document.getElementById("center-column");
        const urlParts = window.location.href.split("/").filter(part => part !== "");
        
        if (urlParts.length > 2) {
            const topic = urlParts[urlParts.length - 1]
            moreDataPath = `${TEST_BASE_PATH}/get_more_news/${topic}&${offset}`;
            
        } else {
            moreDataPath = `${TEST_BASE_PATH}/get_more_news/${offset}`;
        }
        fetch(moreDataPath)
            .then(resp => resp.json())
            .then(data => data.rows.forEach(dataElem => {
                const n = 1 + document.getElementsByClassName("scrollable-content").length;
                // Append the new news to your content
                const newsElement = document.createElement("div");
                newsElement.id = `news-element-${n}`;
                newsElement.className = "scrollable-content";
                /*
                // Image element
                newsSubElement = createElement("img")
                newsSubElement.className = "news-element img-element";
                newsSubElement.src = dataElem.imageUrl;
                newsSubElement.alt = "Image";
                newsSubElement.style.margin = "auto";
                newsElement.appendChild(newsSubElement)
                // creationdate element
                newsSubElement = createElement("p")
                newsSubElement.className = "news-element publish-date";
                newsSubElement.innerHTML = dataElem.creationDate;
                newsElement.appendChild(newsSubElement)
                // title element
                newsSubElement = createElement("h1");
                newsSubElement.className = "news-element title";
                newsSubElement.innerHTML = dataElem.creationDate;
                newsElement.appendChild(newsSubElement);
                // description element
                newsSubElement = createElement("p");
                newsSubElement.className = "news-element description";
                newsSubElement.innerHTML = dataElem.description;
                newsElement.appendChild(newsSubElement);
                // button element
                newsSubElement = createElement("button");
                newsSubElement.className = "news-element button-expand";
                newsSubElement.style.width = "fit-content";
                newsSubElement.style.margin = "10px";
                newsSubElement.dataToggle = "expandable";
                newsSubElement.innerHTML = "Más";
                newsElement.appendChild(newsSubElement);
                // div element
                newsSubElement = createElement("div");
                newsSubElement.className = "news-element more-expandable";
                newsSubElement.style.display = "none";
                //  body elem
                bodyElem = createElement("p");
                bodyElem.innerHTML = dataElem.articleBody;
                newsSubElement.appendChild(bodyElem);
                newsSubElement.appendChild(bodyElem);
                urlElem = createElement("a");
                urlElem.href = dataElem.url;
                urlElem.target = "blank";
                urlElem.innerHTML = msgOriginalUrl;
                
                newsSubElement.appendChild(bodyElem);
                newsSubElement.appendChild(urlElem);
                newsElement.appendChild(newsSubElement);
                */

                // Create news content here similar to your existing template
                // Example:
            newsElement.innerHTML = `
                <img class="news-element img-element" src="${dataElem.imageUrl}" alt="Image" style="margin:auto;">
                <p class="news-element publish-date">${dataElem.creationDate}</p>
                <h1 class="news-element title">${dataElem.title}</h1>
                <p class="news-element description">${dataElem.description}</p>
                <button class="news-element button-expand" style="width: fit-content; margin: 10px" data-toggle="expandable">Más</button>
                <div class="news-element more-expandable" style="display: none">
                    <p>${dataElem.articleBody}</p>
                    <a href="${dataElem.url}" target="blank">Ir al artículo original</a>
                </div>
                <br>
            `;
        
            // Append the news element to the container
            newsContainer.appendChild(newsElement);
            }))
            .catch(error => {
                isLoading = false;
                console.error("Error loading more news:", error);
            });
            isLoading = false;
            offset += newsContainer.length;
    }
}

// Add a scroll event listener to trigger loading more news when the user scrolls to the bottom
window.addEventListener("scroll", () => {
    if (isAtBottom()) {
        loadMoreNews();
    }
});

function isAtBottom() {
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    const scrollTop = window.scrollY;
  
    // Adjust the threshold value as needed
    const threshold = 100;
  
    return scrollTop + windowHeight >= documentHeight - threshold;
  }

// Initial load
loadMoreNews();
