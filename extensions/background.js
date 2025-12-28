// Background service worker cho extension

chrome.runtime.onInstalled.addListener(() => {
    console.log('Email Phishing Detection extension đã được cài đặt');
});

// Lắng nghe messages từ content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'emailUpdated') {
        // Email đã được cập nhật, có thể thực hiện các hành động khác nếu cần
        console.log('Email đã được cập nhật:', message.email);
    }
    return true;
});

// Lắng nghe khi click vào icon extension
chrome.action.onClicked.addListener((tab) => {
    // Popup sẽ tự động mở khi click vào icon
    // Có thể thêm logic khác nếu cần
});

