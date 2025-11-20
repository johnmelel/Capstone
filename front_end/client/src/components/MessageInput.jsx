import React, { useState, useEffect } from 'react';
import './MessageInput.css';

const MessageInput = ({ onSendMessage, isLoading, selectedPatient, patients }) => {
    const [text, setText] = useState('');

    useEffect(() => {
        if (selectedPatient && patients) {
            const patient = patients.find(p => p.id === selectedPatient);
            if (patient) {
                setText(`@${patient.name}: `);
            }
        }
    }, [selectedPatient, patients]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (text.trim() && !isLoading) {
            onSendMessage(text);
            setText('');
        }
    };

    return (
        <form className="message-input-container" onSubmit={handleSubmit}>
            <div className="input-wrapper">
                <input
                    type="text"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Ask a medical question..."
                    disabled={isLoading}
                    className="message-input"
                />
                <button
                    type="submit"
                    disabled={!text.trim() || isLoading}
                    className="send-button"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="send-icon">
                        <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                    </svg>
                </button>
            </div>
        </form>
    );
};

export default MessageInput;
