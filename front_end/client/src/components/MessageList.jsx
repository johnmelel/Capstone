import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import ReasoningDisplay from './ReasoningDisplay';
import './MessageList.css';

const MessageList = ({ messages, isLoading, onPinMessage, pinnedQuestions }) => {
    const messagesEndRef = useRef(null);
    const isInitialState = messages.length === 1 && messages[0].sender === 'bot';

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        if (!isInitialState) {
            scrollToBottom();
        }
    }, [messages, isLoading, isInitialState]);

    const isMessagePinned = (messageId) => {
        return pinnedQuestions?.some(pq => pq.messageId === messageId);
    };

    return (
        <div className={`message-list ${isInitialState ? 'centered' : ''}`}>
            {messages.map((msg) => (
                <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
                    <div className={`message ${msg.sender} ${msg.isError ? 'error' : ''}`}>
                        {msg.sender === 'user' && (
                            <button
                                className={`pin-message-btn ${isMessagePinned(msg.id) ? 'pinned' : ''}`}
                                onClick={() => {
                                    if (isMessagePinned(msg.id)) {
                                        // Find the pinned question and unpin it
                                        const pinnedQ = pinnedQuestions?.find(pq => pq.messageId === msg.id);
                                        if (pinnedQ && onPinMessage) {
                                            // Call with negative ID to indicate unpinning
                                            onPinMessage(msg.id, true);
                                        }
                                    } else {
                                        onPinMessage(msg.id);
                                    }
                                }}
                                title={isMessagePinned(msg.id) ? "Unpin this question" : "Pin this question"}
                            >
                                ✓✓
                            </button>
                        )}
                        <div className="message-content">
                            <ReactMarkdown>{msg.text}</ReactMarkdown>
                        </div>
                        {msg.sender === 'bot' && (msg.reasoning_steps || msg.structured_source || msg.unstructured_source) && (
                            <ReasoningDisplay
                                steps={msg.reasoning_steps}
                                structured={msg.structured_source}
                                unstructured={msg.unstructured_source}
                            />
                        )}
                    </div>
                </div>
            ))}
            {isLoading && (
                <div className="message-wrapper bot">
                    <div className="message bot loading">
                        <div className="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            )}
            <div ref={messagesEndRef} />
        </div>
    );
};

export default MessageList;
