// Navigation Handling
const navLinks = document.querySelectorAll('.nav-links a');
const profileSection = document.getElementById('profileSection');
const profileBtn = document.getElementById('profileBtn');
const mainContent = document.querySelector('.dashboard-content');

// Profile Image Handling
const profileImageInput = document.getElementById('profileImageInput');
const previewImage = document.getElementById('previewImage');
const profileImage = document.getElementById('profileImage');

navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        navLinks.forEach(link => link.parentElement.classList.remove('active'));
        e.currentTarget.parentElement.classList.add('active');

        // Handle section display
        if (e.currentTarget.getAttribute('href') === '#profile') {
            showProfileSection();
        } else {
            hideProfileSection();
        }
    });
});

// Profile Button Click
profileBtn.addEventListener('click', () => {
    showProfileSection();
});

function showProfileSection() {
    profileSection.style.display = 'block';
    document.querySelector('.quick-actions').style.display = 'none';
    document.querySelector('.recent-activity').style.display = 'none';
}

function hideProfileSection() {
    profileSection.style.display = 'none';
    document.querySelector('.quick-actions').style.display = 'block';
    document.querySelector('.recent-activity').style.display = 'block';
}

// Profile Image Upload
profileImageInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            profileImage.src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});

// Save Profile Changes
document.querySelector('.save-profile-btn').addEventListener('click', () => {
    const formData = {
        fullName: document.getElementById('fullName').value,
        email: document.getElementById('email').value,
        phone: document.getElementById('phone').value,
        language: document.getElementById('language').value,
        bio: document.getElementById('bio').value,
        preferences: {
            emailNotifications: document.getElementById('emailNotifications').checked,
            darkMode: document.getElementById('darkMode').checked,
            autoTranslate: document.getElementById('autoTranslate').checked
        }
    };
    
    // Save to localStorage for demo
    localStorage.setItem('userProfile', JSON.stringify(formData));
    alert('Profile updated successfully!');
});

// Load Profile Data
function loadProfileData() {
    const savedProfile = localStorage.getItem('userProfile');
    if (savedProfile) {
        const profile = JSON.parse(savedProfile);
        document.getElementById('fullName').value = profile.fullName || '';
        document.getElementById('email').value = profile.email || '';
        document.getElementById('phone').value = profile.phone || '';
        document.getElementById('language').value = profile.language || 'en';
        document.getElementById('bio').value = profile.bio || '';
        document.getElementById('emailNotifications').checked = profile.preferences?.emailNotifications || false;
        document.getElementById('darkMode').checked = profile.preferences?.darkMode || false;
        document.getElementById('autoTranslate').checked = profile.preferences?.autoTranslate || false;
    }
}

// Quick Action Buttons
const actionButtons = document.querySelectorAll('.action-btn');

actionButtons.forEach(button => {
    button.addEventListener('click', (e) => {
        const action = e.currentTarget.parentElement.querySelector('h3').textContent;
        switch(action) {
            case 'Speech to Sign':
                startSpeechToSign();
                break;
            case 'Sign to Text':
                startSignToText();
                break;
            case 'Text to Sign':
                startTextToSign();
                break;
        }
    });
});

// Translation Functions (Demo)
function startSpeechToSign() {
    if (!('webkitSpeechRecognition' in window)) {
        alert('Speech recognition is not supported in your browser');
        return;
    }
    alert('Starting speech recognition...');
}

function startSignToText() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Camera access is not supported in your browser');
        return;
    }
    alert('Opening camera for sign language recognition...');
}

function startTextToSign() {
    const text = prompt('Enter the text you want to convert to sign language:');
    if (text) {
        alert(`Converting text to sign language: ${text}`);
    }
}

// Notification Handling
const notificationBtn = document.querySelector('.notification-btn');
let notifications = [
    { id: 1, message: "New feature: Speech to Sign translation is now available!" },
    { id: 2, message: "Your last translation was successful" }
];

notificationBtn.addEventListener('click', () => {
    if (notifications.length > 0) {
        alert(notifications.map(n => n.message).join('\n\n'));
    } else {
        alert('No new notifications');
    }
});

// Search Functionality
const searchInput = document.querySelector('.search-bar input');

searchInput.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    // Add search functionality here
    console.log('Searching for:', searchTerm);
});

// Practice Button Handlers
const practiceButtons = document.querySelectorAll('.practice-btn');

practiceButtons.forEach(button => {
    button.addEventListener('click', (e) => {
        const practice = e.target.parentElement.querySelector('h3').textContent;
        alert(`Starting ${practice}`);
    });
});

// Update Progress (Demo)
function updateProgress() {
    const progressCards = document.querySelectorAll('.progress-card');
    
    // Simulate progress updates
    setInterval(() => {
        progressCards.forEach(card => {
            const value = card.querySelector('p');
            if (value.textContent.includes('days')) {
                const days = parseInt(value.textContent) + 1;
                value.textContent = `${days} days`;
            }
        });
    }, 86400000); // Update daily
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadProfileData();
    updateProgress();
}); 