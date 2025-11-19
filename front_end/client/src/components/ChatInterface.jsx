import React, { useState } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import './ChatInterface.css';

const ChatInterface = () => {
    const [messages, setMessages] = useState([
        { id: 1, text: "Hello! I'm your Medical Assistant. How can I help you today?", sender: 'bot' }
    ]);
    const [isLoading, setIsLoading] = useState(false);

    const handleSendMessage = async (text) => {
        // Add user message
        const userMessage = { id: Date.now(), text, sender: 'user' };
        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);

        try {
            // Call API
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: text }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // Add bot message
            const botMessage = {
                id: Date.now() + 1,
                text: data.answer,
                sender: 'bot',
                reasoning_steps: data.reasoning_steps,
                structured_source: data.structured_source,
                unstructured_source: data.unstructured_source
            };
            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Error:', error);
            const errorMessage = { id: Date.now() + 1, text: "Sorry, I encountered an error processing your request.", sender: 'bot', isError: true };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="chat-interface">
            <header className="chat-header">
                <h1>Medical Assistant</h1>
            </header>
            <MessageList messages={messages} isLoading={isLoading} />
            <MessageInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        </div>
    );
};

export default ChatInterface;
