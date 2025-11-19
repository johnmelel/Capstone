import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import ReasoningDisplay from './ReasoningDisplay';
import './MessageList.css';

const MessageList = ({ messages, isLoading }) => {
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    return (
        <div className="message-list">
            {messages.map((msg) => (
                <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
                    <div className={`message ${msg.sender} ${msg.isError ? 'error' : ''}`}>
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
