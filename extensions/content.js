// Content script để đọc email từ Gmail và Outlook

(function() {
    'use strict';

    let lastEmailContent = null;
    let checkInterval = null;

    // Hàm để trích xuất email từ Gmail
    function extractGmailEmail() {
        try {
            // Tìm email đang được mở trong Gmail
            // Gmail sử dụng các selector khác nhau tùy theo view
            const subjectElement = document.querySelector('h2[data-thread-perm-id]') || 
                                 document.querySelector('h2.hP') ||
                                 document.querySelector('span[data-thread-perm-id]') ||
                                 document.querySelector('.h2');
            
            const bodyElement = document.querySelector('.a3s.aiL') || 
                              document.querySelector('.ii.gt') ||
                              document.querySelector('[role="main"] .Am') ||
                              document.querySelector('.a3s');

            if (!subjectElement || !bodyElement) {
                return null;
            }

            const subject = subjectElement.textContent.trim();
            const body = bodyElement.textContent.trim() || bodyElement.innerText.trim();

            if (!subject && !body) {
                return null;
            }

            return {
                subject: subject,
                body: body,
                source: 'gmail'
            };
        } catch (error) {
            console.error('Lỗi khi trích xuất email Gmail:', error);
            return null;
        }
    }

    // Hàm để trích xuất email từ Outlook
    function extractOutlookEmail() {
        try {
            // Outlook Web App selectors
            const subjectElement = document.querySelector('[aria-label*="Subject"]') ||
                                 document.querySelector('div[role="heading"]') ||
                                 document.querySelector('._lv_1h') ||
                                 document.querySelector('h1');
            
            const bodyElement = document.querySelector('[role="textbox"][aria-label*="Message body"]') ||
                              document.querySelector('div[contenteditable="true"][aria-label*="Message"]') ||
                              document.querySelector('._4utP_ccQ') ||
                              document.querySelector('[data-content="true"]');

            if (!subjectElement && !bodyElement) {
                return null;
            }

            const subject = subjectElement ? subjectElement.textContent.trim() : '';
            const body = bodyElement ? (bodyElement.textContent.trim() || bodyElement.innerText.trim()) : '';

            if (!subject && !body) {
                return null;
            }

            return {
                subject: subject,
                body: body,
                source: 'outlook'
            };
        } catch (error) {
            console.error('Lỗi khi trích xuất email Outlook:', error);
            return null;
        }
    }

    // Hàm để trích xuất email (tự động phát hiện Gmail hoặc Outlook)
    function extractEmail() {
        const url = window.location.href;
        
        if (url.includes('mail.google.com')) {
            return extractGmailEmail();
        } else if (url.includes('outlook.live.com') || url.includes('outlook.office.com')) {
            return extractOutlookEmail();
        }
        
        return null;
    }

    // Hàm để kiểm tra và cập nhật email
    function checkAndUpdateEmail() {
        const email = extractEmail();
        
        if (email) {
            // Tạo hash để so sánh
            const emailHash = JSON.stringify(email);
            
            // Chỉ cập nhật nếu email thay đổi
            if (emailHash !== lastEmailContent) {
                lastEmailContent = emailHash;
                
                // Lưu vào storage
                chrome.storage.local.set({ currentEmail: email }, () => {
                    // Gửi message đến background script
                    chrome.runtime.sendMessage({
                        type: 'emailUpdated',
                        email: email
                    });
                });
            }
        }
    }

    // Bắt đầu theo dõi thay đổi email
    function startMonitoring() {
        // Kiểm tra ngay lập tức
        checkAndUpdateEmail();
        
        // Kiểm tra định kỳ mỗi 2 giây
        if (checkInterval) {
            clearInterval(checkInterval);
        }
        
        checkInterval = setInterval(checkAndUpdateEmail, 2000);
        
        // Lắng nghe thay đổi DOM (Gmail và Outlook sử dụng dynamic content)
        const observer = new MutationObserver(() => {
            checkAndUpdateEmail();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    // Dừng theo dõi
    function stopMonitoring() {
        if (checkInterval) {
            clearInterval(checkInterval);
            checkInterval = null;
        }
    }

    // Bắt đầu theo dõi khi page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', startMonitoring);
    } else {
        startMonitoring();
    }

    // Dọn dẹp khi page unload
    window.addEventListener('beforeunload', stopMonitoring);

    console.log('Email Phishing Detection: Content script đã được tải');
})();

