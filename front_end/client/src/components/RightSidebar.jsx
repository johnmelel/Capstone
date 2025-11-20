import React from 'react';
import ReactMarkdown from 'react-markdown';
import './RightSidebar.css';

const RightSidebar = ({ pinnedAnswers }) => {
    return (
        <div className="right-sidebar">
            <h2 className="sidebar-title">Pinned Questions</h2>

            {pinnedAnswers && pinnedAnswers.length > 0 ? (
                <div className="pinned-answers-list">
                    {pinnedAnswers.map((pa) => (
                        <div key={pa.id} className="pinned-answer-item">
                            <div className="question-title">{pa.question}</div>
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
