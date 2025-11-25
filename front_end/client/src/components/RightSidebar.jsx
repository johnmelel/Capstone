import React from 'react';
import ReactMarkdown from 'react-markdown';
import './RightSidebar.css';

const RightSidebar = ({ pinnedAnswers, pinnedQuestions, onUnpinQuestion }) => {
    return (
        <div className="right-sidebar">
            <h2 className="sidebar-title">Pinned Questions</h2>

            {pinnedAnswers && pinnedAnswers.length > 0 ? (
                <div className="pinned-answers-list">
                    {pinnedAnswers.map((pa) => (
                        <div key={pa.id} className="pinned-answer-item">
                            <div className="pinned-answer-header">
                                <div className="question-title">{pa.question}</div>
                                <button
                                    className="unpin-button"
                                    onClick={() => onUnpinQuestion(pa.questionId)}
                                    title="Unpin this question"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="unpin-icon">
                                        <path d="M16 12V4h1c.55 0 1-.45 1-1s-.45-1-1-1H7c-.55 0-1 .45-1 1s.45 1 1 1h1v8l-2 2v2h5v6l1 1 1-1v-6h5v-2l-2-2z" />
                                    </svg>
                                </button>
                            </div>
                            <div className="answer-content">
                                <ReactMarkdown>{pa.answer}</ReactMarkdown>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="empty-state">
                    <p>Pin questions to see auto-updated answers here.</p>
                    <p className="empty-state-subtitle">Answers refresh each time you open the app.</p>
                </div>
            )}
        </div>
    );
};

export default RightSidebar;
