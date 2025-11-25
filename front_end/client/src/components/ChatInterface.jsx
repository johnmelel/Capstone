import React, { useState, useEffect } from 'react';
import LeftSidebar from './LeftSidebar';
import RightSidebar from './RightSidebar';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import ConfirmModal from './ConfirmModal';
import './ChatInterface.css';

// localStorage keys
const STORAGE_KEYS = {
    CONVERSATIONS: 'iris_ai_conversations',
    PINNED_QUESTIONS: 'iris_ai_pinned_questions',
    PINNED_ANSWERS: 'iris_ai_pinned_answers'
};

const ChatInterface = () => {
    // Load initial state from localStorage
    const loadFromStorage = (key, defaultValue) => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error(`Error loading ${key} from localStorage:`, error);
            return defaultValue;
        }
    };

    const [messages, setMessages] = useState([
        { id: 1, text: "Hello! I'm Iris AI, your Medical Assistant. How can I help you today?", sender: 'bot' }
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedPatients, setSelectedPatients] = useState([]);
    const [leftWidth, setLeftWidth] = useState(22);
    const [rightWidth, setRightWidth] = useState(22);
    const [conversations, setConversations] = useState(() => loadFromStorage(STORAGE_KEYS.CONVERSATIONS, []));
    const [pinnedQuestions, setPinnedQuestions] = useState(() => loadFromStorage(STORAGE_KEYS.PINNED_QUESTIONS, []));
    const [pinnedAnswers, setPinnedAnswers] = useState(() => loadFromStorage(STORAGE_KEYS.PINNED_ANSWERS, []));
    const [currentConversationId, setCurrentConversationId] = useState(null);

    // Confirmation modal state
    const [confirmModal, setConfirmModal] = useState({
        isOpen: false,
        message: '',
        onConfirm: null
    });

    // Mock data for patients
    const patients = [
        { id: 1, name: 'Paul Stevens' },
        { id: 2, name: 'Johnson Matt' },
        { id: 3, name: 'Steve Rock' },
        { id: 4, name: 'John Doe' },
        { id: 5, name: 'Jane Smith' },
        { id: 6, name: 'Bob Johnson' },
        { id: 7, name: 'Alice Brown' },
        { id: 8, name: 'Tom Wilson' },
        { id: 9, name: 'Lily Davis' },
        { id: 10, name: 'Mark Taylor' },
        { id: 11, name: 'Emily White' },
        { id: 12, name: 'David Wilson' },
        { id: 13, name: 'Sarah Johnson' },
        { id: 14, name: 'Michael Brown' },
        { id: 15, name: 'Jessica Davis' },
        { id: 16, name: 'William Wilson' },
        { id: 17, name: 'Olivia Taylor' },
        { id: 18, name: 'James White' },
        { id: 19, name: 'Sophia Davis' },
        { id: 20, name: 'Benjamin Wilson' }
    ];

    // Auto-run pinned questions on component mount
    useEffect(() => {
        const runPinnedQuestions = async () => {
            for (const pq of pinnedQuestions) {
                // Check if we already have an answer for this question TEXT (not ID)
                // This prevents duplicate answers when unpinning and re-pinning
                if (!pinnedAnswers.find(pa => pa.question === pq.text)) {
                    try {
                        const response = await fetch('http://localhost:8000/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ query: pq.text }),
                        });

                        if (response.ok) {
                            const data = await response.json();
                            setPinnedAnswers(prev => [...prev, {
                                id: Date.now(),
                                questionId: pq.id,
                                question: pq.text,
                                answer: data.answer,
                                timestamp: new Date().toISOString()
                            }]);
                        }
                    } catch (error) {
                        console.error('Error fetching pinned question answer:', error);
                    }
                }
            }
        };

        if (pinnedQuestions.length > 0) {
            runPinnedQuestions();
        }
    }, [pinnedQuestions]);

    // Save conversations to localStorage whenever they change
    useEffect(() => {
        console.log('Saving conversations to localStorage:', conversations.length);
        localStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));
    }, [conversations]);

    // Save pinned questions to localStorage whenever they change
    useEffect(() => {
        localStorage.setItem(STORAGE_KEYS.PINNED_QUESTIONS, JSON.stringify(pinnedQuestions));
    }, [pinnedQuestions]);

    // Save pinned answers to localStorage whenever they change
    useEffect(() => {
        localStorage.setItem(STORAGE_KEYS.PINNED_ANSWERS, JSON.stringify(pinnedAnswers));
    }, [pinnedAnswers]);

    const handleSendMessage = async (text) => {
        const userMessage = { id: Date.now(), text, sender: 'user' };
        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);

        try {
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

    // Helper function to save the current conversation
    const saveCurrentConversation = () => {
        console.log('saveCurrentConversation called, messages:', messages.length);

        // Only save if there are user messages (more than just the initial bot greeting)
        if (messages.length > 1) {
            const firstUserMessage = messages.find(m => m.sender === 'user');
            const title = firstUserMessage ?
                (firstUserMessage.text.length > 50 ? firstUserMessage.text.substring(0, 50) + '...' : firstUserMessage.text) :
                'New conversation';

            // Check if this conversation is already saved
            if (currentConversationId) {
                // Update existing conversation
                console.log('Updating existing conversation:', currentConversationId);
                setConversations(prev => prev.map(conv =>
                    conv.id === currentConversationId
                        ? { ...conv, messages: [...messages], timestamp: new Date().toISOString() }
                        : conv
                ));
            } else {
                // Save as new conversation
                const newConversation = {
                    id: Date.now(),
                    title,
                    messages: [...messages],
                    timestamp: new Date().toISOString()
                };

                console.log('Saving new conversation:', newConversation.title);
                setConversations(prev => {
                    const updated = [newConversation, ...prev];
                    console.log('Updated conversations:', updated.length);
                    return updated;
                });

                // Set the current conversation ID so future saves update instead of creating duplicates
                setCurrentConversationId(newConversation.id);
            }
        } else {
            console.log('Not saving - only initial message present');
        }
    };

    const handleNewChat = () => {
        console.log('handleNewChat called');

        // Save current conversation before starting a new one
        saveCurrentConversation();

        // Reset to new chat
        setMessages([{ id: Date.now(), text: "Hello! I'm Iris AI, your Medical Assistant. How can I help you today?", sender: 'bot' }]);
        setSelectedPatient(null);
        setCurrentConversationId(null);
    };

    const handleLoadConversation = (conversationId) => {
        console.log('handleLoadConversation called:', conversationId);

        // Save current conversation before switching
        saveCurrentConversation();

        // Load the selected conversation
        const conversation = conversations.find(c => c.id === conversationId);
        if (conversation) {
            setMessages(conversation.messages);
            setCurrentConversationId(conversationId);
        }
    };

    const handlePatientSelect = (patientId) => {
        setSelectedPatients(prev => {
            if (prev.includes(patientId)) {
                // Remove if already selected
                return prev.filter(id => id !== patientId);
            } else {
                // Add if not selected
                return [...prev, patientId];
            }
        });
    };

    const handleClearConversations = () => {
        setConfirmModal({
            isOpen: true,
            message: 'Are you sure you want to follow up this action? All conversations will be permanently deleted.',
            onConfirm: () => {
                setConversations([]);
                localStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify([]));
                setConfirmModal({ isOpen: false, message: '', onConfirm: null });
            }
        });
    };

    const handlePinMessage = (messageId, shouldUnpin = false) => {
        if (shouldUnpin) {
            // Unpin the message
            setPinnedQuestions(prev => prev.filter(pq => pq.messageId !== messageId));
        } else {
            // Pin the message
            const message = messages.find(m => m.id === messageId && m.sender === 'user');
            if (message && !pinnedQuestions.find(pq => pq.messageId === messageId)) {
                const pinnedQuestion = {
                    id: Date.now(),
                    messageId,
                    text: message.text,
                    timestamp: new Date().toISOString()
                };
                setPinnedQuestions(prev => [...prev, pinnedQuestion]);
            }
        }
    };

    const handleUnpinQuestion = (pinnedId) => {
        setConfirmModal({
            isOpen: true,
            message: 'Are you sure you want to follow up this action? This question will be unpinned.',
            onConfirm: () => {
                // Remove from pinned questions
                setPinnedQuestions(prev => prev.filter(pq => pq.id !== pinnedId));
                // Also remove the associated answer
                setPinnedAnswers(prev => prev.filter(pa => pa.questionId !== pinnedId));
                setConfirmModal({ isOpen: false, message: '', onConfirm: null });
            }
        });
    };

    const handleLeftResize = (e) => {
        e.preventDefault();
        const startX = e.clientX;
        const startWidth = leftWidth;

        const handleMouseMove = (e) => {
            const deltaX = e.clientX - startX;
            const containerWidth = window.innerWidth;
            const deltaPercent = (deltaX / containerWidth) * 100;
            const newWidth = Math.min(40, Math.max(15, startWidth + deltaPercent));
            setLeftWidth(newWidth);
        };

        const handleMouseUp = () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    };

    const handleRightResize = (e) => {
        e.preventDefault();
        const startX = e.clientX;
        const startWidth = rightWidth;

        const handleMouseMove = (e) => {
            const deltaX = startX - e.clientX;
            const containerWidth = window.innerWidth;
            const deltaPercent = (deltaX / containerWidth) * 100;
            const newWidth = Math.min(40, Math.max(15, startWidth + deltaPercent));
            setRightWidth(newWidth);
        };

        const handleMouseUp = () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    };

    return (
        <div className="chat-interface">
            <div className="left-sidebar-container" style={{ width: `${leftWidth}%` }}>
                <LeftSidebar
                    patients={patients}
                    conversations={conversations}
                    selectedPatients={selectedPatients}
                    onPatientSelect={handlePatientSelect}
                    onNewChat={handleNewChat}
                    onLoadConversation={handleLoadConversation}
                    onClearConversations={handleClearConversations}
                />
                <div className="resize-handle resize-handle-right" onMouseDown={handleLeftResize} />
            </div>

            <div className="center-chat-area" style={{ width: `${100 - leftWidth - rightWidth}%` }}>
                <div className="iris-badge">
                    <div className="iris-title">Iris AI</div>
                    <div className="iris-subtitle">
                        <span>a product of</span>
                        <img
                            src="/university-logo.png"
                            alt="University of Chicago"
                            className="university-logo"
                            onError={(e) => {
                                e.target.style.display = 'none';
                                e.target.nextSibling.style.display = 'inline';
                            }}
                        />
                        <span style={{ display: 'none' }}>University of Chicago</span>
                    </div>
                </div>
                <MessageList
                    messages={messages}
                    isLoading={isLoading}
                    onPinMessage={handlePinMessage}
                    pinnedQuestions={pinnedQuestions}
                />
                <MessageInput
                    onSendMessage={handleSendMessage}
                    isLoading={isLoading}
                    selectedPatients={selectedPatients}
                    patients={patients}
                />
            </div>

            <div className="right-sidebar-container" style={{ width: `${rightWidth}%` }}>
                <div className="resize-handle resize-handle-left" onMouseDown={handleRightResize} />
                <RightSidebar
                    pinnedAnswers={pinnedAnswers}
                    pinnedQuestions={pinnedQuestions}
                    onUnpinQuestion={handleUnpinQuestion}
                />
            </div>

            <ConfirmModal
                isOpen={confirmModal.isOpen}
                message={confirmModal.message}
                onConfirm={confirmModal.onConfirm}
                onCancel={() => setConfirmModal({ isOpen: false, message: '', onConfirm: null })}
            />
        </div>
    );
};

export default ChatInterface;
