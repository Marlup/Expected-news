window.addEventListener("beforeunload", function (e) {
    e.preventDefault(); // This is necessary to show the confirmation dialog in most browsers.
    e.returnValue = ''; // Set an empty string to customize the confirmation message.

    // You can customize the confirmation message to inform the user.
    const confirmationMessage = 'Are you sure you want to leave this page?';

    // Some browsers will ignore the custom message and display a default message.
    return confirmationMessage;
});